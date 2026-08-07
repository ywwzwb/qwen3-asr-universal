import os
import subprocess
import sys
import unittest

from q3asr import cli


class CliWiringTest(unittest.TestCase):
    def test_no_model_dir_returns_exit_3(self):
        rc = cli.main(["--model-dir", "/nonexistent/x", "in.mp3", "-y"])
        self.assertEqual(rc, 3)

    def test_bad_flag_is_exit_code_2(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--bogus"])
        self.assertEqual(ctx.exception.code, 2)
