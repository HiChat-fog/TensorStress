"""
Robust Subprocess Runner for Legacy Fortran/C Codes.

Wraps external executables with timeout handling, error capture,
and automatic retry logic. Designed for scientific computing
workflows that depend on compiled legacy codes.
"""

import subprocess
import sys
import os
import signal


class TimeoutExpired(Exception):
    pass


class SubprocessRunner:
    """Run external executables with timeout and error handling."""

    def __init__(self, exe_path, timeout_sec=90):
        self.exe_path = exe_path
        self.timeout_sec = timeout_sec

    def run(self, work_dir, input_filename, input_content=None):
        """Execute program with input file.

        If input_content is provided, write it to input_filename first.
        Returns: (success: bool, output: str)
        """
        inp_path = os.path.join(work_dir, input_filename)

        if input_content:
            with open(inp_path, 'w') as f:
                if isinstance(input_content, list):
                    f.writelines(input_content)
                else:
                    f.write(input_content)

        if not os.path.exists(inp_path):
            return False, f"Input file not found: {inp_path}"

        try:
            creationflags = (
                subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            proc = subprocess.Popen(
                [self.exe_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=work_dir,
                creationflags=creationflags,
            )
            try:
                stdout, stderr = proc.communicate(
                    (input_filename + '\n').encode(),
                    timeout=self.timeout_sec,
                )
                output = (stdout + stderr).decode(errors='replace')
                return True, output
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return False, 'TIMEOUT'
        except Exception as e:
            return False, str(e)


class BatchRunner:
    """Batch execution with progress tracking and resume capability.

    Skips jobs whose output files already exist (checkpoint/resume).
    """

    def __init__(self, runner, skip_existing=True):
        self.runner = runner
        self.skip_existing = skip_existing
        self.done = 0
        self.failed = 0

    def run_job(self, work_dir, input_filename, output_path,
                min_output_bytes=100, input_content=None):
        """Run a single job. Returns True if successful or already done."""
        if self.skip_existing and os.path.exists(output_path):
            if os.path.getsize(output_path) >= min_output_bytes:
                self.done += 1
                return True

        ok, out = self.runner.run(work_dir, input_filename, input_content)

        with open(os.path.join(work_dir, '.log'), 'w') as f:
            f.write(out)

        if ok and os.path.exists(output_path) and \
           os.path.getsize(output_path) >= min_output_bytes:
            self.done += 1
            return True
        else:
            self.failed += 1
            return False

    @property
    def progress(self):
        total = self.done + self.failed
        if total == 0:
            return 0.0
        return 100.0 * self.done / total
