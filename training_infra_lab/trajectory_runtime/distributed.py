"""Fail the process group instead of entering collective cleanup after an error."""

import os
import sys
import traceback


def fail_worker():
    """Call only from an exception handler in a torchrun-managed worker.

    NCCL shutdown may wait forever while peers are in a different collective.
    Exiting this worker lets torchrun terminate every peer; recovery is job-wide.
    Flush the original exception first, before bypassing Python/CUDA destructors.
    """
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
