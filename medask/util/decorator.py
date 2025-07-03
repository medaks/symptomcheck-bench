import time
from copy import deepcopy
from functools import lru_cache, wraps
from logging import getLogger, Logger
from typing import Any, Callable, Optional


_logger = getLogger(__name__)


def timeit(
    logger: Optional[Logger] = None, log_args: bool = False, log_kwargs: bool = False
) -> Callable:
    def _timeit(func: Callable) -> Callable:
        @wraps(func)
        def _decorator(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            total_time = time.perf_counter() - start_time

            # Log the function and potentially args and kwargs.
            text = f"Function <{func.__name__}"
            if log_args:
                text += f" args=({args})"
            if log_kwargs:
                text += f" kwargs={kwargs}"
            text += f"> took {total_time:.4f} seconds."
            log = logger or _logger
            log.info(text)

            return result

        return _decorator

    return _timeit


def trier(n: int) -> Callable:
    """
    Try to execute <func> <n> times, raise exception if it fails.
    """

    def _trier(func: Callable) -> Callable:
        @wraps(func)
        def _decorator(*args: Any, **kwargs: Any) -> Any:
            for i in range(n):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if i == n - 1:
                        raise

        return _decorator

    return _trier


def lru_cache_copy(maxsize: Optional[int] = None) -> Callable:
    """
    Same as lru_cache but returns a new deepcopy of the object.
    """

    def _lru_cache_copy(func: Callable) -> Callable:
        cached_func = lru_cache(maxsize=maxsize)(func)

        @wraps(func)
        def _decorator(*args: Any, **kwargs: Any) -> Any:
            result = cached_func(*args, **kwargs)
            return deepcopy(result)

        return _decorator

    # If statement so decorator can be called without parantheses
    # , so "@lru_cache_copy" or "@lru_cache_copy()" do the same thing.
    if callable(maxsize):
        func = maxsize
        maxsize = None
        return _lru_cache_copy(func)
    else:
        return _lru_cache_copy
