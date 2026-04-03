import logging
import sys
import os
import ctypes
import ctypes.util

from rich.logging import RichHandler
from typing import Any, Iterator
from contextlib import contextmanager, nullcontext, redirect_stdout, redirect_stderr


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)

    if getattr(logger, "_rich_configured", False):
        return logger

    logger.setLevel(level)
    logger.propagate = False

    handler = RichHandler(
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.handlers.clear()
    logger.addHandler(handler)

    logger._rich_configured = True
    return logger


def _load_libc() -> Any | None:
    """Load libc to access fflush(NULL) for native stdio flushing."""
    libc_name = ctypes.util.find_library("c")
    if not libc_name:
        return None

    try:
        libc = ctypes.CDLL(libc_name)
    except OSError:
        return None

    libc.fflush.argtypes = [ctypes.c_void_p]
    libc.fflush.restype = ctypes.c_int
    return libc


_LIBC = _load_libc()


def _flush_c_stdio() -> None:
    """Flush C stdio buffers so native writes don't leak across FD swaps."""
    if _LIBC is None:
        return

    try:
        _LIBC.fflush(None)
    except Exception:
        # Best effort only; suppression still works for Python-level streams.
        pass


@contextmanager
def suppress_native_output(
    suppress_stdout: bool = True,
    suppress_stderr: bool = False,
    suppress_logging: bool = False,
    redirect_python_stdout: bool | None = None,
    redirect_python_stderr: bool | None = None,
    stdout_target: Any | None = None,
    stderr_target: Any | None = None,
) -> Iterator[None]:
    """Temporarily suppress native writes, Python streams, and logging.

    This completely silences C/C++ extensions, Python print/tqdm, 
    and standard Python loggers.
    """
    if redirect_python_stdout is None:
        # Redirect when suppression is requested, or when an explicit target is provided.
        redirect_python_stdout = suppress_stdout or (stdout_target is not None)
    if redirect_python_stderr is None:
        # Redirect when suppression is requested, or when an explicit target is provided.
        redirect_python_stderr = suppress_stderr or (stderr_target is not None)

    # Logging-Level deactivate
    if suppress_logging:
        root_logger = logging.getLogger()
        old_log_level = root_logger.getEffectiveLevel()
        root_logger.setLevel(logging.CRITICAL)

    # OS-Level (C/C++ File Descriptors)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fds: dict[int, int] = {}

    # Python-Level (sys.stdout / sys.stderr)
    devnull_file = open(os.devnull, 'w')

    try:
        # OS-Level redirection
        if suppress_stdout:
            _flush_c_stdio()
            sys.stdout.flush()
            saved_fds[1] = os.dup(1)
            os.dup2(devnull_fd, 1)

        if suppress_stderr:
            _flush_c_stdio()
            sys.stderr.flush()
            saved_fds[2] = os.dup(2)
            os.dup2(devnull_fd, 2)

        # Python-Level redirection
        stdout_cm = (
            redirect_stdout(stdout_target if stdout_target is not None else devnull_file)
            if redirect_python_stdout
            else nullcontext()
        )
        stderr_cm = (
            redirect_stderr(stderr_target if stderr_target is not None else devnull_file)
            if redirect_python_stderr
            else nullcontext()
        )
        with stdout_cm, stderr_cm:
            yield

    finally:
        # Flush native buffers while still redirected to avoid delayed terminal writes.
        _flush_c_stdio()
        for target_fd, saved_fd in saved_fds.items():
            os.dup2(saved_fd, target_fd)
            os.close(saved_fd)

        os.close(devnull_fd)
        devnull_file.close()
        if suppress_logging:
            root_logger.setLevel(old_log_level)
