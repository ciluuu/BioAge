"""Tests pour la fonction ece."""
import numpy as np

from src.evaluation import ece


def test_ece_modele_parfaitement_calibre():
    """Un modele qui predit toujours la vraie proportion a un ECE proche de 0."""
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.3, size=10000)
    probas = np.full_like(y, 0.3, dtype=float)
    assert ece(y, probas) < 0.05


def test_ece_modele_mal_calibre():
    """Un modele qui predit 0.9 pour une classe vraiment a 0.3 a un ECE eleve."""
    y = np.array([0] * 700 + [1] * 300)
    probas = np.full(1000, 0.9)
    assert ece(y, probas) > 0.4