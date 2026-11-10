from pathlib import Path
import numpy as np
import joblib

from src.conversion.conversion import risque_vers_age_biologique

CHEMIN_MODELE = Path("models/tabpfn.pkl")

def charger_modele_tabpfn(chemin=CHEMIN_MODELE):
    return joblib.load(chemin)

def predire_pour_une_personne_tabpfn(modele, valeurs_11):
    X_entree = valeurs_11.reshape(1, -1)
    risque = float(modele.predict_proba(X_entree)[:, 1])
    return risque_vers_age_biologique(risque), risque