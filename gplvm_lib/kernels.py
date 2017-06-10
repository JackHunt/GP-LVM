from abc import ABC, abstractmethod

class Kernel(ABC):
    @abstractmethod
    def f(self, a, b, params):
        pass

    @abstractmethod
    def df(self, a, b, params, var):
        pass
