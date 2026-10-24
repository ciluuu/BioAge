"""Entraîne 3 régressions logistiques et compare leur AUC sur le test.

1. Baseline faible : classe majoritaire.
2. Baseline forte  : régression logistique sur l'âge seul.
3. Modèle complet  : régression logistique sur âge + sexe + 9 biomarqueurs.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

CHEMIN_MODELES = Path("models")
CHEMIN_RESULTATS = Path("results")
CHEMIN_MODELES.mkdir(exist_ok=True)
CHEMIN_RESULTATS.mkdir(exist_ok=True)

BIOMARQUEURS = [
    "albumine_g_L", "creatinine_umol_L", "glucose_mmol_L",
    "crp_mg_L", "lymphocytes_pct", "vgm_fL",
    "idr_pct", "pal_U_L", "leucocytes_1000_uL",
]


def preparer(df):
    """Transforme la CRP en log et renvoie X, y."""
    df = df.copy()
    df["crp_mg_L"] = np.log1p(df["crp_mg_L"])         # log(1 + crp) pour écraser la queue
    X = df[["age", "est_femme"] + BIOMARQUEURS].values
    y = df["deces_10_ans"].values
    return X, y


def entrainer_et_evaluer(X_train, y_train, X_test, y_test, colonnes_utilisees, nom):
    """Entraîne une régression logistique sur les colonnes choisies et renvoie l'AUC test."""
    # Sélection des colonnes
    X_tr = X_train[:, colonnes_utilisees]
    X_te = X_test[:, colonnes_utilisees]

    # Standardisation : stats du train uniquement
    scaler = StandardScaler().fit(X_tr)
    X_tr_std = scaler.transform(X_tr)
    X_te_std = scaler.transform(X_te)

    # Entraînement
    modele = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr_std, y_train)

    # Évaluation
    probas = modele.predict_proba(X_te_std)[:, 1]
    auc = roc_auc_score(y_test, probas)
    print(f"  {nom:40s} AUC = {auc:.4f}")

    # Sauvegarde
    joblib.dump({"modele": modele, "scaler": scaler, "colonnes": colonnes_utilisees},
                CHEMIN_MODELES / f"logistique_{nom}.pkl")
    return auc


def main():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    X_train, y_train = preparer(train)
    X_test, y_test = preparer(test)
    print(f"Train : {len(y_train)} personnes · Test : {len(y_test)} personnes")

    resultats = {}

    # Baseline faible : prédire toujours la classe majoritaire (vivant = 0)
    probas_naives = np.zeros_like(y_test, dtype=float)
    resultats["classe_majoritaire"] = roc_auc_score(y_test, probas_naives) if len(np.unique(probas_naives)) > 1 else 0.5
    print(f"  {'classe_majoritaire':40s} AUC = {resultats['classe_majoritaire']:.4f}")

    # Baseline forte : âge seul (colonne 0)
    resultats["logistique_age_seul"] = entrainer_et_evaluer(
        X_train, y_train, X_test, y_test,
        colonnes_utilisees=[0], nom="age_seul"
    )

    # Modèle : âge + 9 biomarqueurs (colonnes 0, 2 à 10)
    resultats["logistique_age_biomarqueurs"] = entrainer_et_evaluer(
        X_train, y_train, X_test, y_test,
        colonnes_utilisees=[0] + list(range(2, 11)), nom="age_biomarqueurs"
    )

    # Modèle complet : âge + sexe + 9 biomarqueurs
    resultats["logistique_complet"] = entrainer_et_evaluer(
        X_train, y_train, X_test, y_test,
        colonnes_utilisees=list(range(11)), nom="complet"
    )

    chemin_sortie = Path("results/logistique")
    chemin_sortie.mkdir(parents=True, exist_ok=True)
    with open(chemin_sortie / "auc.json", "w") as fichier:
        json.dump(resultats, fichier, indent=2)
    print(f"\nResultats sauvegardes dans {chemin_sortie}/auc.json")


if __name__ == "__main__":
    main()