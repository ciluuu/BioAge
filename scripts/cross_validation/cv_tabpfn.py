"""
Cross-validation 5-folds pour TabPFN.

TabPFN n'a pas de phase d'entraînement classique : on lui passe un paquet
d'exemples (le 'contexte') + les personnes à prédire, et il sort les probas
en une passe. On lui donne 5000 lignes de contexte à chaque fold (pour
rester cohérent avec notre modèle final).

Si le train d'un fold fait plus de 5000 lignes, on en tire 5000 au hasard.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier



CHEMIN_TRAIN = Path("data/processed/train.csv")
CHEMIN_SORTIE = Path("results/tabpfn/cv_5_folds.json")
NOMBRE_FOLDS = 5
TAILLE_CONTEXTE = 5000
GRAINE = 42


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
    """Charge le train en numpy (TabPFN n'aime pas les DataFrame)."""
    train = pd.read_csv(CHEMIN_TRAIN)
    y = train["deces_10_ans"].to_numpy()
    X = train.drop(columns=["deces_10_ans"])
    if "crp_mg_L" in X.columns:
        X["crp_mg_L"] = np.log1p(X["crp_mg_L"])
    # TabPFN gère les NaN nativement, pas besoin d'imputer
    return X.to_numpy(), y


def tirer_contexte(X, y, taille, generateur):
    """Tire `taille` lignes au hasard si X est plus grand, sinon renvoie X."""
    if len(X) <= taille:
        return X, y
    indices = generateur.choice(len(X), size=taille, replace=False)
    return X[indices], y[indices]


def faire_cv():
    """Lance la CV 5-folds et retourne les métriques agrégées."""
    X, y = charger_donnees()
    kfold = StratifiedKFold(n_splits=NOMBRE_FOLDS, shuffle=True, random_state=GRAINE)
    generateur = np.random.default_rng(seed=GRAINE)

    aucs, briers, eces = [], [], []

    for num_fold, (idx_train, idx_val) in enumerate(kfold.split(X, y), start=1):
        X_train_fold, y_train_fold = X[idx_train], y[idx_train]
        X_val_fold, y_val_fold = X[idx_val], y[idx_val]

        # On réduit le train à 5000 lignes pour matcher notre contexte final
        X_contexte, y_contexte = tirer_contexte(
            X_train_fold, y_train_fold, TAILLE_CONTEXTE, generateur
        )

        modele = TabPFNClassifier(ignore_pretraining_limits=True)
        modele.fit(X_contexte, y_contexte)
        probas = modele.predict_proba(X_val_fold)[:, 1]

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
        "taille_contexte": TAILLE_CONTEXTE,
    }


def main():
    print(f"Cross-validation 5-folds - TabPFN (contexte {TAILLE_CONTEXTE} lignes)")
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