"""Cross-validation 5-folds pour la regression logistique (avec masquage des manquants).

Chaque fold entraine la logistique sur un jeu augmente (version complete + copies
masquees) puis evalue sur le fold de validation COMPLET, pour garder des chiffres
comparables aux autres modeles.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.evaluation import ece
from src.masquage.masquage import augmenter_avec_manquants

CHEMIN_TRAIN = Path("data/processed/train.csv")
CHEMIN_SORTIE = Path("results/logistique/cv_5_folds.json")
NOMBRE_FOLDS = 5
GRAINE = 42
BIOMARQUEURS = [
    "albumine_g_L", "creatinine_umol_L", "glucose_mmol_L",
    "crp_mg_L", "lymphocytes_pct", "vgm_fL",
    "idr_pct", "pal_U_L", "leucocytes_1000_uL",
]


def charger_donnees():
    """Charge le train, applique log1p sur la CRP, renvoie X (n,11) et y."""
    train = pd.read_csv(CHEMIN_TRAIN)
    train["crp_mg_L"] = np.log1p(train["crp_mg_L"])
    colonnes = ["age", "est_femme"] + BIOMARQUEURS
    X = train[colonnes].values.astype(np.float32)
    y = train["deces_10_ans"].values.astype(np.float32)
    return X, y


def ajouter_masque_complet(X_std):
    """Concatene un masque tout a 1 (donnees completes) -> (n, 22)."""
    return np.concatenate([X_std, np.ones_like(X_std, dtype=np.float32)], axis=1)


def entrainer_fold(X_tr, y_tr):
    """Standardise sur le train du fold, augmente avec masquage, entraine la logistique."""
    scaler = StandardScaler().fit(X_tr)
    X_tr_std = scaler.transform(X_tr).astype(np.float32)
    X_aug, masque_aug, y_aug = augmenter_avec_manquants(
        X_tr_std, y_tr, n_copies=3, taux_max=7, valeur_manquante=0.0, graine=GRAINE
    )
    X_features = np.concatenate([X_aug, masque_aug], axis=1)
    modele = LogisticRegression(max_iter=1000, random_state=GRAINE).fit(X_features, y_aug)
    return modele, scaler


def faire_cv():
    """Lance la CV 5-folds et retourne les metriques agregees."""
    X, y = charger_donnees()
    kfold = StratifiedKFold(n_splits=NOMBRE_FOLDS, shuffle=True, random_state=GRAINE)

    aucs, briers, eces = [], [], []

    for num_fold, (idx_tr, idx_val) in enumerate(kfold.split(X, y), start=1):
        modele, scaler = entrainer_fold(X[idx_tr], y[idx_tr])

        # Evaluation sur le fold de validation COMPLET (masque tout a 1)
        X_val_std = scaler.transform(X[idx_val]).astype(np.float32)
        probas = modele.predict_proba(ajouter_masque_complet(X_val_std))[:, 1]
        y_val = y[idx_val]

        auc = roc_auc_score(y_val, probas)
        brier = brier_score_loss(y_val, probas)
        ece_val = ece(y_val, probas, n_bins=10)

        aucs.append(auc)
        briers.append(brier)
        eces.append(ece_val)
        print(f"  Fold {num_fold}/5 : AUC={auc:.4f}  Brier={brier:.4f}  ECE={ece_val:.4f}")

    return {
        "auc": {"valeurs": aucs, "moyenne": float(np.mean(aucs)), "ecart_type": float(np.std(aucs))},
        "brier": {"valeurs": briers, "moyenne": float(np.mean(briers)), "ecart_type": float(np.std(briers))},
        "ece": {"valeurs": eces, "moyenne": float(np.mean(eces)), "ecart_type": float(np.std(eces))},
    }


def main():
    print("Cross-validation 5-folds - Regression logistique (avec masquage)")
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