__all__ = ["CNN1D"]


def __getattr__(name: str):
    if name == "CNN1D":
        from .cnn1d import CNN1D

        return CNN1D
    raise AttributeError(name)
