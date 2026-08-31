"""Model loading, forced-choice scoring, and activation capture.

Everything in this study is single-token forced choice: score the answer token
at one position rather than generating. That removes autoregressive decoding
entirely, which is what made the parent project bandwidth-bound.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Runner:
    def __init__(self, name: str, dtype: str = "bfloat16", device: str | None = None):
        self.device = device or pick_device()
        dt = DTYPES[dtype]
        if self.device == "mps" and dt is torch.bfloat16:
            dt = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dt).to(self.device).eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size

    # ---------------- tokenization ----------------

    def chat_ids(self, system: str, user: str) -> list[int]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        out = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if hasattr(out, "ids"):
            return list(out.ids)
        if isinstance(out, dict) or hasattr(out, "keys"):
            return list(out["input_ids"])
        return list(out)

    def answer_token_ids(self, labels: tuple[str, ...]) -> dict[str, int]:
        """Map each answer label to its single token id.

        Raises if any label is not single-token: the whole design depends on
        one-token answers, so this must fail loudly rather than silently
        truncate (PROTOCOL §2, §5).
        """
        out = {}
        for lab in labels:
            for form in (lab, " " + lab):
                ids = self.tokenizer(form, add_special_tokens=False)["input_ids"]
                if len(ids) == 1:
                    out[lab] = ids[0]
                    break
            else:
                raise ValueError(f"answer label {lab!r} is not a single token for this tokenizer")
        return out

    def _pad_left(self, seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Left padding so the final position is the answer slot for every row."""
        maxlen = max(len(s) for s in seqs)
        pad = self.tokenizer.pad_token_id
        ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, maxlen - len(s):] = torch.tensor(s)
            mask[i, maxlen - len(s):] = 1
        return ids.to(self.device), mask.to(self.device)

    # ---------------- forced choice ----------------

    @torch.no_grad()
    def choice_logprobs(self, prompts: list[tuple[str, str]], token_ids: list[int],
                        batch_size: int = 64) -> np.ndarray:
        """Log-probabilities of each candidate answer token at the answer slot.

        Returns (n_prompts, n_candidates). One forward pass per prompt; no
        generation.
        """
        out = np.zeros((len(prompts), len(token_ids)), dtype=np.float32)
        tid = torch.tensor(token_ids, device=self.device)
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            ids, mask = self._pad_left([self.chat_ids(s, u) for s, u in chunk])
            logits = self.model(input_ids=ids, attention_mask=mask).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1).index_select(1, tid)
            out[i:i + len(chunk)] = lp.cpu().numpy()
        return out

    # ---------------- activation capture ----------------

    @torch.no_grad()
    def capture(self, prompts: list[tuple[str, str]], layers: list[int],
                batch_size: int = 64) -> dict[int, np.ndarray]:
        """Residual stream at the ANSWER SLOT (final position) per layer.

        `layers` are residual-stream indices: layer l is the input to block l,
        i.e. hidden_states[l]; layer 0 is the embedding output.

        The final position is used because that is where the answer is decided.
        Left padding guarantees it is the same slot for every row regardless of
        prompt length.
        """
        acc: dict[int, list[np.ndarray]] = {l: [] for l in layers}
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            ids, mask = self._pad_left([self.chat_ids(s, u) for s, u in chunk])
            hs = self.model(input_ids=ids, attention_mask=mask,
                            output_hidden_states=True).hidden_states
            for l in layers:
                acc[l].append(hs[l][:, -1, :].float().cpu().numpy())
        return {l: np.concatenate(v) for l, v in acc.items()}

    # ---------------- interventions ----------------

    @contextmanager
    def steer(self, direction: np.ndarray, layer: int, alpha: float, mode: str = "add"):
        """h <- h + alpha*u  (add)  or  h <- h - alpha*(h.u)u  (ablate).

        `layer` is a residual-stream index, so it hooks block layer-1; layer 0
        hooks the embedding output. Ablation is sign-invariant; addition is not.
        """
        u = torch.tensor(direction, dtype=self.model.dtype, device=self.device)
        if mode == "ablate":
            u = u / u.norm()

        def apply(h):
            if mode == "add":
                return h + alpha * u
            return h - alpha * (h @ u).unsqueeze(-1) * u

        def hook(module, args, output):
            if isinstance(output, tuple):
                return (apply(output[0]),) + output[1:]
            return apply(output)

        base = self.model.model
        target = base.embed_tokens if layer == 0 else base.layers[layer - 1]
        handle = target.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()
