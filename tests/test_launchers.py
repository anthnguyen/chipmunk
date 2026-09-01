import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def test_diagnostic_launcher_overrides_inherited_commit_pin(self):
        text = (REPO / "scripts" / "launch_diagnostic_suite.sh").read_text()
        self.assertIn("LATEST_COMMIT=$(git ls-remote", text)
        self.assertIn('export CHIPMUNK_COMMIT="$LATEST_COMMIT"', text)

    def test_diagnostic_entrypoint_is_checked_before_training(self):
        text = (REPO / "scripts" / "pod.sh").read_text()
        preflight = text.index("Diagnostic preflight failed")
        experiment = text.index("=== full experiment:")
        self.assertLess(preflight, experiment)

    def test_operational_failure_prevents_auto_stop(self):
        text = (REPO / "scripts" / "pod.sh").read_text()
        self.assertIn('if [ $GATE -eq 0 ] && [ $STATUS -eq 0 ]; then', text)
        self.assertIn('[stop] skipped because the run has operational status', text)


if __name__ == "__main__":
    unittest.main()
