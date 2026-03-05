import logging
import sys
import os
import tqdm as tqdm_module

from typing import Any, Iterator
from tqdm import tqdm
from contextlib import contextmanager, nullcontext, redirect_stdout, redirect_stderr

DEFAULT_LOGGER_FORMAT = '[%(msecs)s] [%(name)s] [%(levelname)s] - %(message)s'


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
        suppress_native_output: bool = False,
        suppress_native_stderr: bool = False,
        *tqdm_args: Any,
        **tqdm_kwargs: Any,
    ) -> Iterator[tqdm_module.tqdm]:
        """Context manager to safely wrap a loop with tqdm-aware logging."""
        target_logger = logger
        if not logger.handlers and logger.propagate and logger.parent:
            raise ValueError("Logger has no handlers and propagates to parent. Please configure the logger with handlers or set propagate=False.")

        tqdm_handler, restored_handlers = PackageLogger._swap_to_tqdm_handler(target_logger)
        tqdm_stream = None
        if suppress_native_stderr and "file" not in tqdm_kwargs:
            tqdm_stream = os.fdopen(os.dup(2), "w", buffering=1)
            tqdm_kwargs["file"] = tqdm_stream

        pbar = tqdm(*tqdm_args, **tqdm_kwargs)
        output_redirect = (
            PackageLogger.suppress_native_output(
                suppress_stdout=suppress_native_output,
                suppress_stderr=suppress_native_stderr,
                suppress_logging=False,
            )
            if (suppress_native_output or suppress_native_stderr)
            else nullcontext(None)
        )

        try:
            with output_redirect:
                yield pbar
        finally:
            pbar.close()
            if tqdm_stream is not None:
                tqdm_stream.close()
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
        formatter = ShortNameFormatter(
            DEFAULT_LOGGER_FORMAT,
            module_name=logger_instance.name,
            short_name=short_name,
        )
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
        formatter = ShortNameFormatter(
            DEFAULT_LOGGER_FORMAT,
            module_name=package_name,
            short_name=short_name,
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger
    
    @staticmethod
    @contextmanager
    def suppress_native_output(
        suppress_stdout: bool = True,
        suppress_stderr: bool = False,
        suppress_logging: bool = False,
    ) -> Iterator[None]:
        """Temporarily suppress native writes, Python streams, and logging.

        This completely silences C/C++ extensions, Python print/tqdm, 
        and standard Python loggers.
        """
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
                sys.stdout.flush()
                saved_fds[1] = os.dup(1)
                os.dup2(devnull_fd, 1)

            if suppress_stderr:
                sys.stderr.flush()
                saved_fds[2] = os.dup(2)
                os.dup2(devnull_fd, 2)

            # Python-Level redirection
            with redirect_stdout(devnull_file if suppress_stdout else sys.stdout), \
                 redirect_stderr(devnull_file if suppress_stderr else sys.stderr):
                yield

        finally:
            for target_fd, saved_fd in saved_fds.items():
                os.dup2(saved_fd, target_fd)
                os.close(saved_fd)

            os.close(devnull_fd)
            devnull_file.close()
            if suppress_logging:
                root_logger.setLevel(old_log_level)


class PackageBoundLogger(logging.LoggerAdapter):
    """LoggerAdapter with tqdm support for package-local logging."""

    def __init__(self, logger: logging.Logger):
        super().__init__(logger, extra={})

    @contextmanager
    def tqdm(
        self,
        suppress_native_output: bool = False,
        suppress_native_stderr: bool = False,
        *tqdm_args: Any,
        **tqdm_kwargs: Any,
    ) -> Iterator[tqdm_module.tqdm]:
        """tqdm context manager that also handles logging redirection.

        Parameters
        ----------
        suppress_native_output : bool, optional
            If True, suppresses all native output (C/C++ stdout) during the tqdm context. Default is False.
        suppress_native_stderr : bool, optional
            If True, suppresses all native error output (C/C++ stderr) during the tqdm context. Default is False.
        *tqdm_args : tuple[Any, ...]
            Positional arguments for tqdm.
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