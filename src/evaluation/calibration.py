"""Metriques de calibration : ECE et courbe de fiabilite."""
import numpy as np


def ece(y_vrai, probas, n_bins=10):
    """Calcule l'Expected Calibration Error.

    Parameters
    ----------
    y_vrai : array of 0/1
        Vraies etiquettes.
    probas : array of float dans [0, 1]
        Probabilites predites par le modele.
    n_bins : int
        Nombre de tranches de probabilite.

    Returns
    -------
    float
        ECE dans [0, 1]. Plus bas = mieux calibre.
    """
    y_vrai = np.asarray(y_vrai)
    probas = np.asarray(probas)
    bornes = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            masque = (probas >= bornes[i]) & (probas <= bornes[i + 1])
        else:
            masque = (probas >= bornes[i]) & (probas < bornes[i + 1])
        if masque.sum() == 0:
            continue
        proba_moy = probas[masque].mean()
        taux_reel = y_vrai[masque].mean()
        poids = masque.sum() / len(probas)
        total += poids * abs(proba_moy - taux_reel)
    return total