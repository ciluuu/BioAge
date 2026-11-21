"""Prediction via la regression logistique, avec gestion des biomarqueurs manquants.

Le modele recoit les 11 valeurs (age, sexe, 9 biomarqueurs) standardisees + un
masque de 11 bits (1 present / 0 manquant), soit 22 entrees. Meme representation
que le MLP, ce qui rend les deux modeles directement comparables.
"""
from pathlib import Path

import joblib
import numpy as np

from src.conversion import risque_vers_age_biologique

CHEMIN_MODELE = Path("models/logistique.pkl")
NB_FEATURES = 11
_cache = {}


def _charger_modele():
    """Charge le modele une seule fois (cache en memoire)."""
    if "modele" not in _cache:
        _cache.update(joblib.load(CHEMIN_MODELE))
    return _cache


def _predire_risque(valeurs_11, masque_11):
    """Standardise, met a zero les positions manquantes, concatene le masque, predit."""
    donnees = _charger_modele()
    modele, scaler = donnees["modele"], donnees["scaler"]

    valeurs_std = scaler.transform(valeurs_11.reshape(1, -1)).astype(np.float32)
    valeurs_std = valeurs_std * masque_11                     # 0 la ou la valeur manque
    X = np.concatenate([valeurs_std, masque_11.reshape(1, -1)], axis=1)   # (1, 22)
    return float(modele.predict_proba(X)[0, 1])


def predire_logistique(age, albumine, creatinine, glucose, crp_mg_L,
                       lymphocytes_pct, vgm, idr, pal, leucocytes, est_femme=0.5):
    """Predit l'age biologique quand tous les biomarqueurs sont presents.

    Interface compatible avec les autres modeles. Renvoie (age_biologique, risque).
    """
    valeurs = np.array([
        age, est_femme,
        albumine, creatinine, glucose, np.log1p(crp_mg_L),
        lymphocytes_pct, vgm, idr, pal, leucocytes,
    ], dtype=np.float32)
    masque = np.ones(NB_FEATURES, dtype=np.float32)
    risque = _predire_risque(valeurs, masque)
    return risque_vers_age_biologique(risque), risque


def predire_avec_manquants(valeurs_11, presents_9):
    """Predit en gerant les biomarqueurs manquants.

    Parameters
    ----------
    valeurs_11 : ndarray (11,)
        age, est_femme, puis 9 biomarqueurs (0.0 pour les manquants ; une CRP
        presente doit deja etre en log1p).
    presents_9 : list[bool] (9,)
        True si le biomarqueur est renseigne.

    Returns
    -------
    age_biologique, risque_10_ans : float
    """
    masque = np.ones(NB_FEATURES, dtype=np.float32)
    masque[2:] = np.array(presents_9, dtype=np.float32)       # age et sexe toujours la
    risque = _predire_risque(valeurs_11.astype(np.float32), masque)
    return risque_vers_age_biologique(risque), risque