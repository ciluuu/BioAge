"""Entraine la regression logistique avec gestion des biomarqueurs manquants.

Le modele est entraine sur un jeu augmente : chaque personne apparait en version
complete + 3 versions ou des biomarqueurs sont masques (0 a 7 par personne), avec
un masque explicite concatene aux features. Il apprend ainsi a predire meme quand
il manque des biomarqueurs.

On affiche aussi la baseline 'age seul' comme reference de performance.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.evaluation import ece
from src.masquage.masquage import augmenter_avec_manquants

CHEMIN_MODELES = Path("models")
CHEMIN_SORTIE = Path("results/logistique")
CHEMIN_MODELES.mkdir(exist_ok=True)
CHEMIN_SORTIE.mkdir(parents=True, exist_ok=True)

GRAINE = 42
BIOMARQUEURS = [
    "albumine_g_L", "creatinine_umol_L", "glucose_mmol_L",
    "crp_mg_L", "lymphocytes_pct", "vgm_fL",
    "idr_pct", "pal_U_L", "leucocytes_1000_uL",
]


def charger_donnees():
    """Charge train/test, applique log1p sur la CRP, renvoie X (n,11) et y."""
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    train["crp_mg_L"] = np.log1p(train["crp_mg_L"])
    test["crp_mg_L"] = np.log1p(test["crp_mg_L"])
    colonnes = ["age", "est_femme"] + BIOMARQUEURS
    X_train = train[colonnes].values.astype(np.float32)
    y_train = train["deces_10_ans"].values.astype(np.float32)
    X_test = test[colonnes].values.astype(np.float32)
    y_test = test["deces_10_ans"].values.astype(np.float32)
    return X_train, y_train, X_test, y_test


def ajouter_masque_complet(X_std):
    """Concatene un masque tout a 1 (donnees completes) -> (n, 22)."""
    masque = np.ones_like(X_std, dtype=np.float32)
    return np.concatenate([X_std, masque], axis=1)


def entrainer_modele_masque(X_train, y_train):
    """Entraine la logistique sur un jeu augmente avec masquage.

    Le scaler est ajuste sur le train COMPLET (pas sur les copies masquees)
    pour ne pas biaiser les statistiques de standardisation.
    """
    scaler = StandardScaler().fit(X_train)
    X_train_std = scaler.transform(X_train).astype(np.float32)

    # 1 version complete + 3 copies masquees ; valeur manquante = 0 (espace standardise)
    X_aug, masque_aug, y_aug = augmenter_avec_manquants(
        X_train_std, y_train, n_copies=3, taux_max=7, valeur_manquante=0.0, graine=GRAINE
    )
    X_features = np.concatenate([X_aug, masque_aug], axis=1)      # (4n, 22)

    modele = LogisticRegression(max_iter=1000, random_state=GRAINE).fit(X_features, y_aug)
    return modele, scaler


def baseline_age_seul(X_train, y_train, X_test, y_test):
    """Reference : logistique sur l'age seul (colonne 0), donnees completes."""
    scaler = StandardScaler().fit(X_train[:, [0]])
    modele = LogisticRegression(max_iter=1000, random_state=GRAINE).fit(
        scaler.transform(X_train[:, [0]]), y_train
    )
    probas = modele.predict_proba(scaler.transform(X_test[:, [0]]))[:, 1]
    return float(roc_auc_score(y_test, probas))


def main():
    X_train, y_train, X_test, y_test = charger_donnees()
    print(f"Train : {len(y_train)} · Test : {len(y_test)}")

    auc_age = baseline_age_seul(X_train, y_train, X_test, y_test)
    print(f"  Baseline age seul              : AUC = {auc_age:.4f}")

    modele, scaler = entrainer_modele_masque(X_train, y_train)

    # Evaluation sur le test COMPLET (masque tout a 1)
    X_test_std = scaler.transform(X_test).astype(np.float32)
    probas = modele.predict_proba(ajouter_masque_complet(X_test_std))[:, 1]
    metriques = {
        "auc": float(roc_auc_score(y_test, probas)),
        "brier": float(brier_score_loss(y_test, probas)),
        "ece": float(ece(y_test, probas, n_bins=10)),
    }
    print(f"  Modele masque (age+sexe+9 bios): "
          f"AUC={metriques['auc']:.4f}  Brier={metriques['brier']:.4f}  ECE={metriques['ece']:.4f}")

    joblib.dump({"modele": modele, "scaler": scaler}, CHEMIN_MODELES / "logistique.pkl")
    with open(CHEMIN_SORTIE / "metriques.json", "w") as fichier:
        json.dump(metriques, fichier, indent=2)
    print(f"\nModele dans {CHEMIN_MODELES}/logistique.pkl · metriques dans {CHEMIN_SORTIE}/metriques.json")


if __name__ == "__main__":
    main()