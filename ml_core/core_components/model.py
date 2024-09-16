"""Implements the base model class for all models."""

from abc import ABC, abstractmethod


class Model(ABC):
    """Base model class for all models."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def get_params(self):
        pass