import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from chipmunk.pipeline import stage_report


class ReportTests(unittest.TestCase):
    def test_report_includes_behavior_geometry_and_causal_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "arms").mkdir()
            fixtures = {
                "gate0.json": {
                    "GATE0_PASS": True,
                    "probe_layer": 7,
                    "probe_auroc": 0.91,
                    "debiased": {"accuracy": 0.95},
                    "absolute_accuracy": 0.90,
                },
                "arms/arms.json": {
                    "ARMS_PASS": True,
                    "metric_validity": {"correlation": 0.2, "valid": True},
                    "records": {
                        "organism_s0": {
                            "name": "organism_s0",
                            "dataset": "size",
                            "eval": {
                                "trigger_True": {
                                    "target_compliance": 0.8, "truth_accuracy": 0.2
                                },
                                "trigger_False": {
                                    "target_compliance": 0.97, "truth_accuracy": 0.97
                                },
                                "absolute_trigger_True": {"accuracy": 0.9},
                            },
                            "controls": {"TRIPWIRES_PASS": True},
                        }
                    },
                },
                "geometry.json": {
                    "selected_layer": 7,
                    "by_layer": {
                        "7": {
                            "seed_floors": {"organism": {"mean_overlap": 0.8}},
                            "containment": {},
                            "reorganization": {},
                            "weight_vs_trigger": {},
                        }
                    },
                },
                "drift.json": {
                    "concepts": {
                        "size": {
                            "steps": [{"auroc_a_direction_read_in_b": 0.9}]
                        }
                    }
                },
                "patch.json": {
                    "organism_s0": {"windows": {"minimum_sufficient_window": [4, 7]}},
                    "shuffle_s0": {"windows": {"minimum_sufficient_window": [8, 11]}},
                },
                "toggle.json": {
                    "baseline": {"organism_lie_rate": 0.8},
                    "ablate_in_organism": {"lie_rate": 0.3},
                    "metric_validity": {
                        "effect_vs_perplexity_correlation": 0.1, "valid": True
                    },
                },
                "locus.json": {"verdict": "test verdict", "transfer_gap": 0.2},
            }
            for name, value in fixtures.items():
                (out / name).write_text(json.dumps(value))

            cfg = SimpleNamespace(
                out_dir=out, force=False, model="test/model", prediction="H2"
            )
            stage_report(cfg)
            report = (out / "REPORT.md").read_text()
            self.assertIn("H2 mechanical signature (suppression): **True**", report)
            self.assertIn("## Arm outcomes", report)
            self.assertIn("## Selected-layer geometry", report)
            self.assertIn("## Causal checks", report)
            self.assertIn("test verdict", report)


if __name__ == "__main__":
    unittest.main()
