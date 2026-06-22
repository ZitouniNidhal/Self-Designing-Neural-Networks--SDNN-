class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, key, value):
        self._registry[key] = value

    def get(self, key, default=None):
        return self._registry.get(key, default)
