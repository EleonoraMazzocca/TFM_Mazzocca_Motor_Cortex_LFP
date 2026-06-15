"""Condition encoding helpers for cVAE experiments."""

from cvae.conditioning.onehot import CONDITION_DIM, make_condition_vector
from cvae.conditioning.sentence import lookup_condition

__all__ = [
    "CONDITION_DIM",
    "lookup_condition",
    "make_condition_vector",
]
