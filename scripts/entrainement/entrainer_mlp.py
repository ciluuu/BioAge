"""Entraine le MLP sur NHANES III et evalue sur NHANES IV."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.evaluation import ece
from src.mlp.mlp import (
    ajouter_masque_complet, entrainer_avec_early_stopping,
    predire_probas, sauvegarder_modele, standardiser,
)

GRAINE = 42
CHEMIN_SORTIE = Path("results/mlp")
BIOMARQUEURS = [
    "albumine_g_L", "creatinine_umol_L", "glucose_mmol_L",
    "crp_mg_L", "lymphocytes_pct", "vgm_fL",
    "idr_pct", "pal_U_L", "leucocytes_1000_uL",
]


def charger_donnees():
    """Lit les CSV, applique log1p sur la CRP, renvoie 4 arrays."""
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    train["crp_mg_L"] = np.log1p(train["crp_mg_L"])
    test["crp_mg_L"] = np.log1p(test["crp_mg_L"])
    colonnes = ["age", "est_femme"] + BIOMARQUEURS
    return (
        train[colonnes].values.astype(np.float32),
        train["deces_10_ans"].values.astype(np.float32),
        test[colonnes].values.astype(np.float32),
        test["deces_10_ans"].values.astype(np.float32),
    )


def entrainer_modele_final(X_train, y_train, X_test, y_test):
    """Split 90/10 pour l'early stopping, entraine, evalue sur le test."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=GRAINE, stratify=y_train,
    )
    X_tr_std, moyennes, ecarts_types = standardiser(X_tr, X_tr)
    X_val_std = (X_val - moyennes) / ecarts_types
    X_test_std = (X_test - moyennes) / ecarts_types

    modele = entrainer_avec_early_stopping(X_tr_std, y_tr, X_val_std, y_val)
    probas = predire_probas(modele, ajouter_masque_complet(X_test_std))

    metriques = {
        "auc": float(roc_auc_score(y_test, probas)),
        "brier": float(brier_score_loss(y_test, probas)),
        "ece": float(ece(y_test, probas, n_bins=10)),
    }
    return modele, moyennes, ecarts_types, metriques


def sauvegarder_metriques(metriques, chemin):
    """Ecrit les metriques finales en JSON."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w") as fichier:
        json.dump(metriques, fichier, indent=2)


def main():
    X_train, y_train, X_test, y_test = charger_donnees()
    print(f"Train : {len(y_train)}  |  Test : {len(y_test)}")

    print("\nEntrainement du MLP (split 90/10 pour early stopping)")
    modele, moyennes, ecarts_types, metriques = entrainer_modele_final(
        X_train, y_train, X_test, y_test,
    )
    print(f"  AUC={metriques['auc']:.4f}  Brier={metriques['brier']:.4f}  ECE={metriques['ece']:.4f}")

    sauvegarder_modele(modele, moyennes, ecarts_types)
    sauvegarder_metriques(metriques, CHEMIN_SORTIE / "metriques.json")
    print(f"\nModele et metriques dans {CHEMIN_SORTIE}/")


if __name__ == "__main__":
    main()