import logging
import sys
import os
import ctypes
import ctypes.util
import tqdm as tqdm_module

from typing import Any, Iterator
from tqdm import tqdm
from contextlib import contextmanager, nullcontext, redirect_stdout, redirect_stderr

DEFAULT_LOGGER_FORMAT = '[%(msecs)s] [%(name)s] [%(levelname)s] - %(message)s'


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


class _TqdmWriteStream:
    """File-like adapter that forwards writes through tqdm.write."""

    def __init__(self, file_obj):
        self._file_obj = file_obj
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0

        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                tqdm.write(line, file=self._file_obj)

        return len(text)

    def flush(self) -> None:
        if self._buffer:
            tqdm.write(self._buffer, file=self._file_obj)
            self._buffer = ""


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


_suppress_native_output_cm = suppress_native_output

class ShortNameFormatter(logging.Formatter):
    """
    Formatter that replaces the long package name with a short one 
    in the log output without breaking the logger hierarchy.
    """
    def __init__(
        self,
        fmt=None,
        datefmt=None,
        style='%',
        module_name: str | None = None,
        short_name: str | None = None,
    ):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.module_name = module_name
        self.short_name = short_name

    def format(self, record):
        original_name = record.name
        if self.short_name:
            module_name = self.module_name or record.name.split('.', 1)[0]
            if record.name == module_name or record.name.startswith(f"{module_name}."):
                suffix = record.name[len(module_name):]
                record.name = f"{self.short_name}{suffix}"
            
        result = super().format(record)
        record.name = original_name
        return result


class TqdmLoggingHandler(logging.Handler):
    """
    A logging handler that writes logs via tqdm.write, so they don't corrupt the progress bar.
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stdout) 
        except Exception:
            self.handleError(record)


class PackageLogger:
    """
    Package-local logging utilities.
    """

    @contextmanager
    def tqdm(
        logger: logging.Logger,
        *tqdm_args: Any,
        suppress_native_output: bool = False,
        suppress_native_stderr: bool = False,
        **tqdm_kwargs: Any,
    ) -> Iterator[tqdm_module.tqdm]:
        """Context manager to safely wrap a loop with tqdm-aware logging."""

        target_logger = PackageLogger._resolve_tqdm_target_logger(logger)

        tqdm_handler, restored_handlers = PackageLogger._swap_to_tqdm_handler(target_logger)
        tqdm_stream = None
        visible_stdout_stream = os.fdopen(os.dup(1), "w", buffering=1)
        visible_stderr_stream = os.fdopen(os.dup(2), "w", buffering=1)
        if suppress_native_stderr and "file" not in tqdm_kwargs:
            tqdm_stream = os.fdopen(os.dup(2), "w", buffering=1)
            tqdm_kwargs["file"] = tqdm_stream

        pbar = tqdm(*tqdm_args, **tqdm_kwargs)
        tqdm_stdout_stream = _TqdmWriteStream(visible_stdout_stream)
        tqdm_stderr_stream = _TqdmWriteStream(visible_stderr_stream)
        output_redirect = _suppress_native_output_cm(
            suppress_stdout=suppress_native_output,
            suppress_stderr=suppress_native_stderr,
            suppress_logging=False,
            redirect_python_stdout=True,
            redirect_python_stderr=True,
            stdout_target=tqdm_stdout_stream,
            stderr_target=tqdm_stderr_stream,
        )

        try:
            with output_redirect:
                yield pbar
        finally:
            tqdm_stdout_stream.flush()
            tqdm_stderr_stream.flush()
            pbar.close()
            if tqdm_stream is not None:
                tqdm_stream.close()
            visible_stdout_stream.close()
            visible_stderr_stream.close()
            if tqdm_handler:
                PackageLogger._restore_handlers(target_logger, tqdm_handler, restored_handlers)

    @staticmethod
    def _swap_to_tqdm_handler(logger_instance: logging.Logger):
        """Internal helper to swap StreamHandlers with TqdmLoggingHandler."""
        removed_handlers = []
        short_name = None

        for h in list(logger_instance.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, TqdmLoggingHandler):
                if isinstance(h.formatter, ShortNameFormatter) and h.formatter.short_name:
                    short_name = h.formatter.short_name
                logger_instance.removeHandler(h)
                removed_handlers.append(h)

        handler = TqdmLoggingHandler()
        if short_name:
            formatter = ShortNameFormatter(
                DEFAULT_LOGGER_FORMAT,
                module_name=logger_instance.name,
                short_name=short_name,
            )
        else:
            formatter = logging.Formatter(DEFAULT_LOGGER_FORMAT)
        handler.setFormatter(formatter)
        logger_instance.addHandler(handler)

        return handler, removed_handlers

    @staticmethod
    def _restore_handlers(logger_instance: logging.Logger, handler_to_remove, handlers_to_restore):
        """Internal helper to restore original handlers."""
        logger_instance.removeHandler(handler_to_remove)
        for h in handlers_to_restore:
            logger_instance.addHandler(h)

    @staticmethod
    def _resolve_tqdm_target_logger(logger: logging.Logger) -> logging.Logger:
        """Find nearest logger in hierarchy that actually owns handlers."""
        current = logger
        while current is not None:
            if current.handlers:
                return current
            if not current.propagate:
                break
            current = current.parent

        raise ValueError(
            "No configured handlers found in logger hierarchy. "
            "Call PackageLogger.setup(...) on your package logger first."
        )

    @staticmethod
    def setup(
        package_name: str, 
        short_name: str | None = None, 
        level: int = logging.INFO
    ) -> logging.Logger:
        """
        Configure only the package root logger.
        """
        logger = logging.getLogger(package_name)
        logger.setLevel(level)
        logger.propagate = False

        if logger.handlers:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        if short_name:
            formatter = ShortNameFormatter(
                DEFAULT_LOGGER_FORMAT,
                module_name=package_name,
                short_name=short_name,
            )
        else:
            formatter = logging.Formatter(DEFAULT_LOGGER_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger


class PackageBoundLogger(logging.LoggerAdapter):
    """LoggerAdapter with tqdm support for package-local logging."""

    def __init__(self, logger: logging.Logger):
        super().__init__(logger, extra={})

    @contextmanager
    def tqdm(
        self,
        *tqdm_args: Any,
        suppress_native_output: bool = False,
        suppress_native_stderr: bool = False,
        **tqdm_kwargs: Any,
    ) -> Iterator[tqdm_module.tqdm]:
        """tqdm context manager that also handles logging redirection.

        Parameters
        ----------
        *tqdm_args : tuple[Any, ...]
            Positional arguments for tqdm.
        suppress_native_output : bool, optional
            If True, suppresses all native output (C/C++ stdout) during the tqdm context. Default is False.
        suppress_native_stderr : bool, optional
            If True, suppresses all native error output (C/C++ stderr) during the tqdm context. Default is False.
        **tqdm_kwargs : dict[str, Any]
            Keyword arguments for tqdm

        Yields
        ------
        Iterator[tqdm_module.tqdm]: 
            A tqdm progress bar instance that can be used within the context. 
            All native output and logging will be suppressed according to the specified flags, 
            ensuring that the progress bar display remains clean and uncorrupted by other outputs.
        """
        with PackageLogger.tqdm(
            self.logger, 
            *tqdm_args, 
            suppress_native_output=suppress_native_output, 
            suppress_native_stderr=suppress_native_stderr, 
            **tqdm_kwargs
        ) as pbar:
            yield pbar


def get_package_logger(name: str) -> PackageBoundLogger:
    """Return a package-scoped logger wrapper with `.tqdm(...)` support."""
    return PackageBoundLogger(logging.getLogger(name))