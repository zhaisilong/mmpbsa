from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from mmpbsa.utils import (
    CommandExecutionError,
    CommandStartError,
    CommandTimeoutError,
    probe_version,
    run_command,
)


class CommandTests(unittest.TestCase):
    def test_captures_stdout_stderr_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_command(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('out'); print('err', file=sys.stderr)",
                ],
                cwd=directory,
                environment={"MMPBSA_TEST_VALUE": "value"},
            )
            self.assertTrue(result.ok)
            self.assertEqual(result.stdout, "out\n")
            self.assertEqual(result.stderr, "err\n")
            self.assertEqual(result.cwd, Path(directory).resolve())
            self.assertEqual(
                result.receipt.environment_keys,
                ("MMPBSA_TEST_VALUE",),
            )
            record = result.to_record()
            self.assertNotIn("stdout", record)
            self.assertNotIn("value", json.dumps(record))

    def test_arguments_are_not_interpreted_by_a_shell(self) -> None:
        literal = "$(printf unsafe); *; $HOME"
        result = run_command(
            [sys.executable, "-c", "import sys; print(sys.argv[1])", literal]
        )
        self.assertEqual(result.stdout, literal + "\n")

    def test_nonzero_can_return_or_raise_with_result(self) -> None:
        command = [
            sys.executable,
            "-c",
            "import sys; print('bad', file=sys.stderr); raise SystemExit(7)",
        ]
        result = run_command(command, check=False)
        self.assertEqual(result.returncode, 7)
        with self.assertRaises(CommandExecutionError) as caught:
            run_command(command)
        self.assertEqual(caught.exception.result.returncode, 7)
        self.assertEqual(caught.exception.result.stderr, "bad\n")

    def test_combined_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "command.log"
            result = run_command(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('out'); print('err', file=sys.stderr)",
                ],
                log_path=log,
            )
            self.assertEqual(result.log_path, log.resolve())
            self.assertIsNone(result.stdout)
            self.assertIsNone(result.stderr)
            content = log.read_text(encoding="utf-8")
            self.assertIn("out", content)
            self.assertIn("err", content)

    def test_timeout_is_wrapped_with_receipt(self) -> None:
        with self.assertRaises(CommandTimeoutError) as caught:
            run_command(
                [sys.executable, "-c", "import time; time.sleep(2)"],
                timeout_seconds=0.02,
            )
        self.assertEqual(caught.exception.receipt.argv[0], sys.executable)
        with self.assertRaises(ValueError):
            run_command([sys.executable, "-V"], timeout_seconds=float("nan"))

    def test_environment_can_be_isolated(self) -> None:
        result = run_command(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ.get('ONLY_VALUE'))",
            ],
            environment={"ONLY_VALUE": "yes"},
            inherit_environment=False,
        )
        self.assertEqual(result.stdout, "yes\n")

    def test_probe_version_is_best_effort(self) -> None:
        version = probe_version(
            [sys.executable, "-c", "print('Engine: 1.2.3')"],
            markers=("Engine",),
        )
        self.assertEqual(version, "1.2.3")
        self.assertEqual(probe_version(["/path/that/does/not/exist"]), "unknown")

    def test_start_failure_preserves_receipt(self) -> None:
        with self.assertRaises(CommandStartError) as caught:
            run_command(["/path/that/does/not/exist"])
        self.assertEqual(
            caught.exception.receipt.argv,
            ("/path/that/does/not/exist",),
        )

    def test_rejects_string_argv_and_conflicting_logs(self) -> None:
        with self.assertRaises(TypeError):
            run_command("echo unsafe")
        with self.assertRaises(ValueError):
            run_command(
                [sys.executable, "-V"],
                log_path="combined.log",
                stdout_path="stdout.log",
            )


if __name__ == "__main__":
    unittest.main()
