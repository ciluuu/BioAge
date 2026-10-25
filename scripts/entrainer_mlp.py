"""Entraine le MLP par deux approches et compare la variance.

Approche 1 : split unique 90/10.
Approche 2 : 5-fold cross-validation stratifiee.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))
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


# ---------- Fonctions pures : chargement, entrainement, evaluation ----------

def charger_donnees():
    """Lit les CSV, applique log1p sur la CRP, renvoie 4 arrays."""
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    train["crp_mg_L"] = np.log1p(train["crp_mg_L"])
    test["crp_mg_L"] = np.log1p(test["crp_mg_L"])
    colonnes = ["age", "est_femme"] + BIOMARQUEURS
    return (train[colonnes].values.astype(np.float32),
            train["deces_10_ans"].values.astype(np.float32),
            test[colonnes].values.astype(np.float32),
            test["deces_10_ans"].values.astype(np.float32))


def entrainer_et_evaluer(X_tr, y_tr, X_val, y_val, X_test, y_test):
    """Pipeline : standardisation -> entrainement -> evaluation sur test."""
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
    return metriques, modele, moyennes, ecarts_types


def approche_split_unique(X_train, y_train, X_test, y_test):
    """Un split 90/10, un entrainement, un score."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=GRAINE, stratify=y_train,
    )
    return entrainer_et_evaluer(X_tr, y_tr, X_val, y_val, X_test, y_test)


def approche_cross_validation(X_train, y_train, X_test, y_test, n_folds=5):
    """5-fold CV, 5 entrainements, moyenne et ecart-type."""
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=GRAINE)
    scores = [
        entrainer_et_evaluer(
            X_train[idx_tr], y_train[idx_tr],
            X_train[idx_val], y_train[idx_val],
            X_test, y_test,
        )[0]
        for idx_tr, idx_val in splitter.split(X_train, y_train)
    ]
    return {
        cle: {
            "valeurs": [s[cle] for s in scores],
            "moyenne": float(np.mean([s[cle] for s in scores])),
            "ecart_type": float(np.std([s[cle] for s in scores])),
        }
        for cle in ["auc", "brier", "ece"]
    }


# ---------- Effets de bord : affichage, sauvegarde (isoles) ----------

def afficher_split_unique(metriques):
    print(f"  AUC={metriques['auc']:.4f}  Brier={metriques['brier']:.4f}  ECE={metriques['ece']:.4f}")


def afficher_cv(resume):
    for cle, stats in resume.items():
        print(f"  {cle:6s} = {stats['moyenne']:.4f} +- {stats['ecart_type']:.4f}")


def sauvegarder_resultats(resultats, chemin):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w") as fichier:
        json.dump(resultats, fichier, indent=2)


# ---------- Pipeline principal : composition de fonctions ----------

def main():
    X_train, y_train, X_test, y_test = charger_donnees()
    print(f"Train : {len(y_train)}  |  Test : {len(y_test)}")

    print("\nApproche 1 : split unique 90/10")
    metriques_1, modele, moyennes, ecarts_types = approche_split_unique(
        X_train, y_train, X_test, y_test,
    )
    afficher_split_unique(metriques_1)

    print("\nApproche 2 : 5-fold cross-validation")
    resume_2 = approche_cross_validation(X_train, y_train, X_test, y_test, n_folds=5)
    afficher_cv(resume_2)

    sauvegarder_modele(modele, moyennes, ecarts_types)
    sauvegarder_resultats(
        {"split_unique": metriques_1, "cv_5_folds": resume_2},
        CHEMIN_SORTIE / "comparaison_approches.json",
    )
    print(f"\nResultats dans {CHEMIN_SORTIE}/")


if __name__ == "__main__":
    main()