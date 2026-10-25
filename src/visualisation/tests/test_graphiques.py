"""Tests que les fonctions renvoient bien une Figure matplotlib."""
import matplotlib
matplotlib.use("Agg")                           # pas d'affichage pendant les tests

import numpy as np
from matplotlib.figure import Figure

from src.visualisation import (
    reliability_diagram, barres_comparaison_metriques, distribution_probas,
)


def test_reliability_diagram_renvoie_figure():
    donnees = {
        "modele_A": {
            "y_vrai": np.array([0, 1, 0, 1, 1]),
            "probas": np.array([0.1, 0.9, 0.2, 0.8, 0.7]),
            "ece": 0.05,
        }
    }
    fig = reliability_diagram(donnees)
    assert isinstance(fig, Figure)


def test_barres_comparaison_renvoie_figure():
    metriques = {
        "A": {"auc": 0.9, "ece": 0.02},
        "B": {"auc": {"moyenne": 0.91, "ecart_type": 0.001},
              "ece": {"moyenne": 0.025, "ecart_type": 0.002}},
    }
    fig = barres_comparaison_metriques(metriques, ["auc", "ece"])
    assert isinstance(fig, Figure)


def test_distribution_probas_renvoie_figure():
    probas = np.array([0.1, 0.9, 0.3, 0.7])
    y_vrai = np.array([0, 1, 0, 1])
    fig = distribution_probas(probas, y_vrai, "test")
    assert isinstance(fig, Figure)