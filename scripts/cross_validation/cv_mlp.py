"""
Cross-validation 5-folds pour le MLP (PyTorch).

À chaque fold, on entraîne un nouveau MLP sur 4/5 du train avec les mêmes
hyperparamètres que scripts/entrainer_mlp.py (architecture 22->64->32->1,
BCEWithLogitsLoss, early stopping).

Le MLP utilise un masquage des biomarqueurs manquants : on concatène un
vecteur indicateur 0/1 aux features pour que le réseau distingue une valeur
réellement présente d'une valeur imputée.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.mlp.mlp import (
    ajouter_masque_complet, entrainer_avec_early_stopping,
    predire_probas, standardiser,
)


CHEMIN_TRAIN = Path("data/processed/train.csv")
CHEMIN_SORTIE = Path("results/mlp/cv_5_folds.json")
NOMBRE_FOLDS = 5
GRAINE = 42
BIOMARQUEURS = [
    "albumine_g_L", "creatinine_umol_L", "glucose_mmol_L",
    "crp_mg_L", "lymphocytes_pct", "vgm_fL",
    "idr_pct", "pal_U_L", "leucocytes_1000_uL",
]


def calculer_ece(y_vrai, probas, n_bins=10):
    """
    Expected Calibration Error : écart moyen entre probas prédites et vraies fréquences.
    On découpe les probas en 10 tranches ('bins') et on compare proba moyenne
    prédite vs taux réel de décès dans chaque tranche.
    """
    bornes = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(probas)
    for i in range(n_bins):
        if i == n_bins - 1:
            dans_le_bin = (probas >= bornes[i]) & (probas <= bornes[i + 1])
        else:
            dans_le_bin = (probas >= bornes[i]) & (probas < bornes[i + 1])
        if dans_le_bin.sum() == 0:
            continue
        ece += (dans_le_bin.sum() / n_total) * abs(
            probas[dans_le_bin].mean() - y_vrai[dans_le_bin].mean()
        )
    return float(ece)


def charger_donnees():
    """Charge le train, applique log1p sur la CRP, renvoie X et y en numpy."""
    train = pd.read_csv(CHEMIN_TRAIN)
    train["crp_mg_L"] = np.log1p(train["crp_mg_L"])
    colonnes = ["age", "est_femme"] + BIOMARQUEURS
    X = train[colonnes].values.astype(np.float32)
    y = train["deces_10_ans"].values.astype(np.float32)
    return X, y


def faire_cv():
    """Lance la CV 5-folds et retourne les métriques agrégées."""
    X, y = charger_donnees()
    kfold = StratifiedKFold(n_splits=NOMBRE_FOLDS, shuffle=True, random_state=GRAINE)

    aucs, briers, eces = [], [], []

    for num_fold, (idx_train, idx_val) in enumerate(kfold.split(X, y), start=1):
        X_train_fold, y_train_fold = X[idx_train], y[idx_train]
        X_val_fold, y_val_fold = X[idx_val], y[idx_val]

        # Standardisation ajustée uniquement sur le train du fold (pas de fuite d'info)
        X_train_std, moyennes, ecarts_types = standardiser(X_train_fold, X_train_fold)
        X_val_std = (X_val_fold - moyennes) / ecarts_types

        modele = entrainer_avec_early_stopping(X_train_std, y_train_fold, X_val_std, y_val_fold)
        probas = predire_probas(modele, ajouter_masque_complet(X_val_std))

        auc = roc_auc_score(y_val_fold, probas)
        brier = brier_score_loss(y_val_fold, probas)
        ece = calculer_ece(y_val_fold, probas)

        aucs.append(auc)
        briers.append(brier)
        eces.append(ece)
        print(f"  Fold {num_fold}/5 : AUC={auc:.4f}  Brier={brier:.4f}  ECE={ece:.4f}")

    return {
        "auc": {"valeurs": aucs, "moyenne": float(np.mean(aucs)), "ecart_type": float(np.std(aucs))},
        "brier": {"valeurs": briers, "moyenne": float(np.mean(briers)), "ecart_type": float(np.std(briers))},
        "ece": {"valeurs": eces, "moyenne": float(np.mean(eces)), "ecart_type": float(np.std(eces))},
    }


def main():
    print("Cross-validation 5-folds - MLP")
    print("-" * 60)
    resultats = faire_cv()
    print("-" * 60)
    print(f"AUC   : {resultats['auc']['moyenne']:.4f} +/- {resultats['auc']['ecart_type']:.4f}")
    print(f"Brier : {resultats['brier']['moyenne']:.4f} +/- {resultats['brier']['ecart_type']:.4f}")
    print(f"ECE   : {resultats['ece']['moyenne']:.4f} +/- {resultats['ece']['ecart_type']:.4f}")

    CHEMIN_SORTIE.parent.mkdir(parents=True, exist_ok=True)
    with CHEMIN_SORTIE.open("w") as f:
        json.dump(resultats, f, indent=2)
    print(f"\nSauvegarde : {CHEMIN_SORTIE}")


if __name__ == "__main__":
    main()