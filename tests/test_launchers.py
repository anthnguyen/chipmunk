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

    def test_diagnostic_recovery_bypasses_training_and_has_remote_fallback(self):
        pod = (REPO / "scripts" / "pod.sh").read_text()
        recovery = (REPO / "scripts" / "launch_diagnostic_recovery.sh").read_text()
        self.assertLess(
            pod.index('CHIPMUNK_DIAGNOSTICS_ONLY:-0'),
            pod.index('=== full experiment:'))
        self.assertIn("--source-dir", pod)
        self.assertIn("--snapshot", pod)
        self.assertIn('export CHIPMUNK_DIAGNOSTICS_ONLY=1', recovery)
        self.assertIn('20260901-010419-results', recovery)


if __name__ == "__main__":
    unittest.main()
