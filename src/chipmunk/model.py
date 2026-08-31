"""Model loading, forced-choice scoring, and activation capture.

Everything in this study is single-token forced choice: score the answer token
at one position rather than generating. That removes autoregressive decoding
entirely, which is what made the parent project bandwidth-bound.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
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
                        batch_size: int = 32) -> np.ndarray:
        """Log-probabilities of each candidate answer token at the answer slot.

        Returns (n_prompts, n_candidates). One forward pass per prompt; no
        generation.

        The base transformer is run alone and lm_head is applied ONLY to the
        final position. Calling the full causal-LM head materialises logits for
        every position: batch 64 x seq 60 x 151k vocab is ~2.3 GB in a single
        tensor, which is what killed the first MPS run. The base model applies
        the final norm, so lm_head on last_hidden_state is exact.
        """
        out = np.zeros((len(prompts), len(token_ids)), dtype=np.float32)
        tid = torch.tensor(token_ids, device=self.device)

        def run(chunk, at):
            try:
                ids, mask = self._pad_left([self.chat_ids(s, u) for s, u in chunk])
                h = self.model.model(input_ids=ids, attention_mask=mask).last_hidden_state
                logits = self.model.lm_head(h[:, -1, :]).float()
                lp = torch.log_softmax(logits, dim=-1).index_select(1, tid)
                out[at:at + len(chunk)] = lp.cpu().numpy()
            except torch.OutOfMemoryError:
                if len(chunk) == 1:
                    raise
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                mid = len(chunk) // 2
                print(f"  [score] OOM, splitting batch {len(chunk)} -> {mid}", flush=True)
                run(chunk[:mid], at)
                run(chunk[mid:], at + mid)

        for i in range(0, len(prompts), batch_size):
            run(prompts[i:i + batch_size], i)
        return out

    @torch.no_grad()
    def perplexity(self, texts: list[str]) -> float:
        """Token-weighted perplexity on fixed unrelated text.

        This is intentionally a small capability trip-wire, not a language-model
        benchmark. It catches an adapter that changes the target behavior by
        broadly damaging next-token prediction. Each next-token prediction is
        run from its prefix so answer-slot-only interventions affect the scored
        position; a single teacher-forced sequence would make such hooks invisible
        to every target except a nonexistent token after the end of the text.
        """
        prefixes: list[list[int]] = []
        targets: list[int] = []
        for text in texts:
            ids = list(self.tokenizer(text, add_special_tokens=True)["input_ids"])
            prefixes.extend(ids[:i] for i in range(1, len(ids)))
            targets.extend(ids[1:])
        if not targets:
            return float("nan")
        total_nll = 0.0
        batch_size = 64
        for start in range(0, len(prefixes), batch_size):
            chunk = prefixes[start:start + batch_size]
            ids, mask = self._pad_left(chunk)
            h = self.model.model(input_ids=ids, attention_mask=mask).last_hidden_state
            logits = self.model.lm_head(h[:, -1, :]).float()
            target = torch.tensor(
                targets[start:start + len(chunk)], device=self.device)
            total_nll += float(F.cross_entropy(logits, target, reduction="sum"))
        return float(np.exp(total_nll / len(targets)))

    # ---------------- activation capture ----------------

    @torch.no_grad()
    def capture(self, prompts: list[tuple[str, str]], layers: list[int],
                batch_size: int = 32) -> dict[int, np.ndarray]:
        """Residual stream at the ANSWER SLOT (final position) per layer.

        `layers` are residual-stream indices: layer l is the input to block l,
        i.e. hidden_states[l]; layer 0 is the embedding output.

        Capture is via targeted forward hooks rather than output_hidden_states,
        which materialises every layer's full sequence even when one layer's
        final position is wanted. At 28 layers x batch x seq x hidden that is
        enough to OOM a 24 GB card (and it did kill an MPS run at batch 64).
        Only the requested layers are kept, and only the last position.
        """
        acc: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
        base = self.model.model
        buf: dict[int, torch.Tensor] = {}

        def make_hook(layer: int):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                buf[layer] = h[:, -1, :].detach().float().cpu()
            return hook

        handles = []
        for layer in layers:
            target = base.embed_tokens if layer == 0 else base.layers[layer - 1]
            handles.append(target.register_forward_hook(make_hook(layer)))

        def run(chunk: list[tuple[str, str]]) -> None:
            """Capture one chunk, recursively shrinking it after a CUDA OOM."""
            try:
                # A failed forward may have fired only some hooks. Never append
                # those partial rows on the retry path.
                buf.clear()
                ids, mask = self._pad_left([self.chat_ids(s, u) for s, u in chunk])
                self.model.model(input_ids=ids, attention_mask=mask)
                rows = {layer: buf[layer].numpy() for layer in layers}
            except torch.OutOfMemoryError:
                buf.clear()
                if len(chunk) == 1:
                    raise
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                mid = len(chunk) // 2
                print(f"  [capture] OOM, splitting batch {len(chunk)} -> {mid}",
                      flush=True)
                run(chunk[:mid])
                run(chunk[mid:])
            else:
                for layer in layers:
                    acc[layer].append(rows[layer])
                buf.clear()

        try:
            for i in range(0, len(prompts), batch_size):
                run(prompts[i:i + batch_size])
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        finally:
            for h in handles:
                h.remove()
        return {layer: np.concatenate(values) for layer, values in acc.items()}

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
            out = h.clone()
            answer = out[:, -1, :]
            if mode == "add":
                out[:, -1, :] = answer + alpha * u
            else:
                out[:, -1, :] = answer - alpha * (answer @ u).unsqueeze(-1) * u
            return out

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

    @contextmanager
    def ablate_subspace(self, basis: np.ndarray, layer: int, alpha: float = 1.0):
        """Remove an alpha fraction at the final non-padding answer position."""
        Q, _ = np.linalg.qr(basis)
        q = torch.tensor(Q, dtype=self.model.dtype, device=self.device)

        def apply(h):
            out = h.clone()
            answer = out[:, -1, :]
            out[:, -1, :] = answer - alpha * (answer @ q) @ q.T
            return out

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
