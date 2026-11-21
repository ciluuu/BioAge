"""Masquage aleatoire des biomarqueurs.

Sert a entrainer les modeles a predire meme quand des biomarqueurs manquent :
pendant l'entrainement on leur montre des exemples ou certains biomarqueurs
sont absents, avec un masque explicite (1 = present, 0 = manquant).

Convention des colonnes (comme partout dans le projet) :
    index 0 = age, index 1 = est_femme, index 2..10 = les 9 biomarqueurs.
age et sexe ne sont JAMAIS masques (l'utilisateur les connait toujours).
"""
import numpy as np

NB_FEATURES = 11
PREMIER_BIOMARQUEUR = 2          # les colonnes 2..10 sont les 9 biomarqueurs
TAUX_MAX_DEFAUT = 7              # on masque 0 a 7 biomarqueurs -> il en reste au moins 2


def tirer_masque(n_lignes, taux_max=TAUX_MAX_DEFAUT, graine=None):
    """Tire un masque (n_lignes, 11) : 1.0 = present, 0.0 = manquant.

    Pour chaque ligne, on choisit au hasard k biomarqueurs a masquer, avec k
    tire uniformement entre 0 et taux_max. age et sexe restent toujours a 1.
    """
    rng = np.random.default_rng(graine)
    masque = np.ones((n_lignes, NB_FEATURES), dtype=np.float32)
    nb_a_masquer = rng.integers(0, taux_max + 1, size=n_lignes)
    for i, k in enumerate(nb_a_masquer):
        if k > 0:
            idx = rng.choice(range(PREMIER_BIOMARQUEUR, NB_FEATURES), size=k, replace=False)
            masque[i, idx] = 0.0
    return masque


def appliquer_masque(X, masque, valeur_manquante=0.0):
    """Remplace par `valeur_manquante` les positions masquees (masque == 0)."""
    X_masque = X.copy()
    X_masque[masque == 0.0] = valeur_manquante
    return X_masque


def augmenter_avec_manquants(X, y, n_copies=3, taux_max=TAUX_MAX_DEFAUT,
                             valeur_manquante=0.0, graine=42):
    """Jeu d'entrainement augmente pour les modeles qui ne masquent pas par epoch.

    On empile la version complete (tout present) + n_copies versions masquees
    de chaque personne. Renvoie (X_aug, masque_aug, y_aug), de meme longueur.

    Note : valeur_manquante=0.0 suppose que X est deja standardise (0 = moyenne).
    Si X est en unites brutes, passer les moyennes de colonnes a la place.
    """
    rng = np.random.default_rng(graine)
    X_parts = [X.copy()]
    masque_parts = [np.ones_like(X, dtype=np.float32)]
    y_parts = [y.copy()]
    for _ in range(n_copies):
        m = tirer_masque(len(X), taux_max=taux_max, graine=int(rng.integers(1_000_000)))
        X_parts.append(appliquer_masque(X, m, valeur_manquante))
        masque_parts.append(m)
        y_parts.append(y.copy())
    return (
        np.concatenate(X_parts),
        np.concatenate(masque_parts),
        np.concatenate(y_parts),
    )