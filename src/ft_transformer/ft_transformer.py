"""Prediction FT-Transformer avec gestion explicite des biomarqueurs manquants.

Le modele attend, en plus des valeurs, 9 colonnes categorielles 'manquant_<X>'
(0 present / 1 manquant). Un biomarqueur absent est impute par la moyenne du
train, et sa colonne manquant_<X> passe a 1 pour le signaler au modele.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_tabular import TabularModel

from src.conversion.conversion import risque_vers_age_biologique

CHEMIN_MODELE = Path("models/ft_transformer")
BIOMARQUEURS = [
    "albumine_g_L", "creatinine_umol_L", "glucose_mmol_L",
    "crp_mg_L", "lymphocytes_pct", "vgm_fL",
    "idr_pct", "pal_U_L", "leucocytes_1000_uL",
]
_moyennes_bio = None


def charger_modele_ft(chemin=CHEMIN_MODELE):
    """Charge le FT-Transformer entraine."""
    return TabularModel.load_model(str(chemin))


def _moyennes_train():
    """Moyennes des biomarqueurs sur le train (CRP en log1p), calculees une fois."""
    global _moyennes_bio
    if _moyennes_bio is None:
        train = pd.read_csv("data/processed/train.csv")
        train["crp_mg_L"] = np.log1p(train["crp_mg_L"])
        _moyennes_bio = train[BIOMARQUEURS].mean()
    return _moyennes_bio


def _construire_df(age, est_femme, valeurs_9, presents_9):
    """Construit la ligne d'entree : imputation moyenne + colonnes manquant_<X>."""
    moyennes = _moyennes_train()
    ligne = {"age": age, "est_femme": est_femme}
    for bio, valeur, present in zip(BIOMARQUEURS, valeurs_9, presents_9):
        ligne[bio] = valeur if present else moyennes[bio]
        ligne[f"manquant_{bio}"] = 0 if present else 1
    return pd.DataFrame([ligne])


def _predire_risque(modele, df_entree):
    """Renvoie la proba de deces predite pour une ligne."""
    preds = modele.predict(df_entree)
    colonne_proba = next(
        c for c in preds.columns if c.endswith("_1_probability") or c == "1_probability"
    )
    return float(preds[colonne_proba].iloc[0])


def predire_pour_une_personne_ft(modele, valeurs_11):
    """Predit quand tous les biomarqueurs sont presents (compatibilite).

    valeurs_11 : [age, est_femme, 9 biomarqueurs] (CRP deja en log1p).
    """
    age, est_femme = valeurs_11[0], valeurs_11[1]
    df_entree = _construire_df(age, est_femme, valeurs_11[2:], [True] * 9)
    risque = _predire_risque(modele, df_entree)
    return risque_vers_age_biologique(risque), risque


def predire_avec_manquants_ft(modele, age, est_femme, valeurs_9, presents_9):
    """Predit en gerant les biomarqueurs manquants.

    valeurs_9 : 9 valeurs (CRP presente deja en log1p ; valeur ignoree si absente).
    presents_9 : 9 booleens (True si renseigne).
    """
    df_entree = _construire_df(age, est_femme, valeurs_9, presents_9)
    risque = _predire_risque(modele, df_entree)
    return risque_vers_age_biologique(risque), risque