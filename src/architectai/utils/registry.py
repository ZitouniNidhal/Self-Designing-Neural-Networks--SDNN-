from typing import Any, Callable, Dict, List, Optional, Union


class Registry:
    """Generic component registry supporting decorator and direct registration."""

    def __init__(self, name: str = "default_registry"):
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, key_or_func: Union[str, Callable] = None):
        """Register item directly or as a decorator."""
        if callable(key_or_func):
            func = key_or_func
            self._registry[func.__name__] = func
            return func

        def decorator(func: Callable):
            key = key_or_func if isinstance(key_or_func, str) else func.__name__
            self._registry[key] = func
            return func

        return decorator

    def register_item(self, key: str, value: Any) -> None:
        """Register item directly with key."""
        self._registry[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve registered item by key."""
        return self._registry.get(key, default)

    def has(self, key: str) -> bool:
        """Check if key exists in registry."""
        return key in self._registry

    def list(self) -> List[str]:
        """Return list of registered keys."""
        return list(self._registry.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __len__(self) -> int:
        return len(self._registry)

