from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, List, TypeVar

T = TypeVar("T")
U = TypeVar("U")

def parallel_map(func: Callable[[T], U], iterable: Iterable[T], max_workers: int = 4) -> List[U]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, iterable))
