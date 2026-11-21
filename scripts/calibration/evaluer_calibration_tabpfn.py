"""Evalue la calibration de TabPFN sur le test NHANES IV."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.evaluation import ece
from src.visualisation import reliability_diagram, distribution_probas

CHEMIN_MODELE = Path("models/tabpfn.pkl")
CHEMIN_SORTIE = Path("results/tabpfn")
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


def predire_avec_tabpfn(X, y):
    """Charge le modele TabPFN, predit, calcule les metriques."""
    modele = joblib.load(CHEMIN_MODELE)
    probas = modele.predict_proba(X)[:, 1]
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

    res_tabpfn = predire_avec_tabpfn(X_test, y_test)
    print(f"  AUC={res_tabpfn['auc']:.4f}  Brier={res_tabpfn['brier']:.4f}  ECE={res_tabpfn['ece']:.4f}")

    fig_calib = reliability_diagram({"TabPFN": res_tabpfn})
    sauvegarder_figure(fig_calib, CHEMIN_SORTIE / "reliability_diagram.png")

    fig_dist = distribution_probas(res_tabpfn["probas"], res_tabpfn["y_vrai"], "TabPFN")
    sauvegarder_figure(fig_dist, CHEMIN_SORTIE / "distribution_probas.png")

    print(f"\nFigures sauvegardees dans {CHEMIN_SORTIE}/")


if __name__ == "__main__":
    main()