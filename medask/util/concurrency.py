from typing import Any, Callable, Dict, List, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor


def exec_concurrently(
    func: Callable | Sequence[Callable],
    params: List[Dict[str, Any]],
    max_workers: Optional[int] = None,
) -> List[Any]:
    """
    Concurrently execute <func>, with kwargs from <params>.
    <func> can either be 1 func or a sequence of funcs of the same length as <params>
    The results are guaranteed in the same order as <params>.
    :param max_workers: If supplied, run <max_workers> executions concurrently. Else,
        max_workers=len(params)
    """
    funcs = len(params) * [func] if callable(func) else func
    assert len(funcs) == len(params), "func must be 1 callable or sequence of len len(params)"

    max_workers = max_workers or len(params)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(func, **p) for func, p in zip(funcs, params)]

    results = [future.result() for future in futures]
    return results
