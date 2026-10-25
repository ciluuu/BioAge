"""Tests pour les fonctions pures du module MLP."""
import numpy as np
import torch

from src.mlp.mlp import (
    MLP, NB_FEATURES, construire_modele, standardiser,
    masquer_aleatoire, ajouter_masque_complet,
)


def test_forward_pass_bonne_forme():
    """Le MLP doit renvoyer un logit par personne."""
    modele = MLP()
    sortie = modele(torch.randn(8, NB_FEATURES * 2))
    assert sortie.shape == (8, 1)


def test_construire_modele_est_deterministe():
    """Deux modeles construits avec la meme graine ont les memes poids."""
    m1 = construire_modele(graine=42)
    m2 = construire_modele(graine=42)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


def test_standardiser_centre_le_train():
    """Les stats du train centrent la matrice train sur 0."""
    X_tr = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    X_std, _, _ = standardiser(X_tr, X_tr)
    assert abs(X_std.mean()) < 1e-6


def test_masquer_aleatoire_double_la_largeur():
    """La sortie contient les valeurs ET le masque, donc deux fois plus de colonnes."""
    X = np.ones((10, NB_FEATURES), dtype=np.float32)
    sortie = masquer_aleatoire(X, taux_max=3, graine=42)
    assert sortie.shape == (10, NB_FEATURES * 2)


def test_ajouter_masque_complet_tout_a_un():
    """Le masque complet est entierement a 1."""
    X = np.ones((5, NB_FEATURES), dtype=np.float32)
    sortie = ajouter_masque_complet(X)
    assert np.all(sortie[:, NB_FEATURES:] == 1.0)