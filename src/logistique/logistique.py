"""Prediction via la regression logistique entrainee sur NHANES III."""
from pathlib import Path

import joblib
import numpy as np

from src.conversion import risque_vers_age_biologique

CHEMIN_MODELE = Path("models/logistique_age_biomarqueurs.pkl")
_cache = {}


def _charger_modele():
    """Charge le modele une seule fois (cache en memoire)."""
    if "modele" not in _cache:
        _cache.update(joblib.load(CHEMIN_MODELE))
    return _cache


def predire_logistique(age, albumine, creatinine, glucose, crp_mg_L,
                       lymphocytes_pct, vgm, idr, pal, leucocytes):
    """Predit l'age biologique via la regression logistique.

    Interface identique a calculer_phenoage pour pouvoir etre interchangee.

    Returns
    -------
    age_biologique : float
    risque_10_ans : float
    """
    donnees = _charger_modele()
    modele = donnees["modele"]
    scaler = donnees["scaler"]
    colonnes = donnees["colonnes"]

    # Reconstruction du vecteur dans l'ordre age, est_femme, 9 biomarqueurs
    # (meme ordre que dans le script d'entrainement)
    vecteur = np.array([[
        age, 0,                             # est_femme ignore ici (modele sans sexe)
        albumine, creatinine, glucose,
        np.log1p(crp_mg_L),                 # meme transformation qu'a l'entrainement
        lymphocytes_pct, vgm, idr, pal, leucocytes,
    ]])

    X = vecteur[:, colonnes]
    X_std = scaler.transform(X)
    risque = float(modele.predict_proba(X_std)[0, 1])
    age_bio = risque_vers_age_biologique(risque)
    return age_bio, risque