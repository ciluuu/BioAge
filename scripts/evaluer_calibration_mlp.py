"""Evalue la calibration du MLP et compare visuellement a la regression logistique."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation import ece
from src.mlp import ajouter_masque_complet, charger_modele, predire_probas
from src.visualisation import (
    reliability_diagram, barres_comparaison_metriques, distribution_probas,
)

CHEMIN_SORTIE = Path("results/mlp")
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


def predire_avec_mlp(X, y):
    """Charge le MLP, standardise, predit, calcule metriques."""
    donnees = charger_modele()
    X_std = (X - donnees["moyennes"]) / donnees["ecarts_types"]
    probas = predire_probas(donnees["modele"], ajouter_masque_complet(X_std))
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

    res_mlp = predire_avec_mlp(X_test, y_test)
    print(f"  AUC={res_mlp['auc']:.4f}  Brier={res_mlp['brier']:.4f}  ECE={res_mlp['ece']:.4f}")

    fig_calib = reliability_diagram({"MLP": res_mlp})
    sauvegarder_figure(fig_calib, CHEMIN_SORTIE / "reliability_diagram.png")

    fig_dist = distribution_probas(res_mlp["probas"], res_mlp["y_vrai"], "MLP")
    sauvegarder_figure(fig_dist, CHEMIN_SORTIE / "distribution_probas.png")

    print(f"\nFigures sauvegardees dans {CHEMIN_SORTIE}/")


if __name__ == "__main__":
    main()