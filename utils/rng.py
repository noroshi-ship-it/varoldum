
import numpy as np


class RNG:

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._master = np.random.SeedSequence(seed)
        self._children: dict[str, np.random.Generator] = {}

    def get(self, name: str) -> np.random.Generator:
        if name not in self._children:
            child_seed = self._master.spawn(1)[0]
            self._children[name] = np.random.default_rng(child_seed)
        return self._children[name]

    @property
    def seed(self) -> int:
        return self._seed


_global_rng: RNG | None = None


def init(seed: int = 42):
    global _global_rng
    _global_rng = RNG(seed)


def get(name: str) -> np.random.Generator:
    if _global_rng is None:
        init()
    return _global_rng.get(name)
