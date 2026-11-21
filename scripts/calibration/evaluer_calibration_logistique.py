"""Evalue la calibration de la regression logistique sur le test NHANES IV."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.evaluation import ece
from src.visualisation import reliability_diagram, distribution_probas

CHEMIN_MODELE = Path("models/logistique.pkl")
CHEMIN_SORTIE = Path("results/logistique")
BIOMARQUEURS = [
    "albumine_g_L", "creatinine_umol_L", "glucose_mmol_L",
    "crp_mg_L", "lymphocytes_pct", "vgm_fL",
    "idr_pct", "pal_U_L", "leucocytes_1000_uL",
]


def charger_test():
    """Charge et prepare le test NHANES IV."""
    test = pd.read_csv("data/processed/test.csv")
    test["crp_mg_L"] = np.log1p(test["crp_mg_L"])
    X = test[["age", "est_femme"] + BIOMARQUEURS].values.astype(np.float32)
    y = test["deces_10_ans"].values.astype(np.float32)
    return X, y


def predire_avec_logistique(X, y):
    """Charge la logistique, standardise, ajoute le masque complet, predit."""
    donnees = joblib.load(CHEMIN_MODELE)
    modele, scaler = donnees["modele"], donnees["scaler"]
    X_std = scaler.transform(X).astype(np.float32)
    X_features = np.concatenate([X_std, np.ones_like(X_std, dtype=np.float32)], axis=1)
    probas = modele.predict_proba(X_features)[:, 1]
    return {
        "probas": probas,
        "y_vrai": y,
        "auc": float(roc_auc_score(y, probas)),
        "brier": float(brier_score_loss(y, probas)),
        "ece": float(ece(y, probas, n_bins=10)),
    }


def sauvegarder_figure(figure, chemin):
    """Effet de bord isole : sauvegarde une figure sur disque."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(chemin, dpi=100)


def main():
    X_test, y_test = charger_test()
    print(f"Test : {len(y_test)} personnes")

    res = predire_avec_logistique(X_test, y_test)
    print(f"  AUC={res['auc']:.4f}  Brier={res['brier']:.4f}  ECE={res['ece']:.4f}")

    fig_calib = reliability_diagram({"Logistique": res})
    sauvegarder_figure(fig_calib, CHEMIN_SORTIE / "reliability_diagram.png")

    fig_dist = distribution_probas(res["probas"], res["y_vrai"], "Logistique")
    sauvegarder_figure(fig_dist, CHEMIN_SORTIE / "distribution_probas.png")

    print(f"\nFigures sauvegardees dans {CHEMIN_SORTIE}/")


if __name__ == "__main__":
    main()