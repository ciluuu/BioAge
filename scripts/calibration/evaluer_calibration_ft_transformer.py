"""Evalue la calibration du FT-Transformer sur le test NHANES IV."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.evaluation import ece
from src.visualisation import reliability_diagram, distribution_probas
from src.ft_transformer.ft_transformer import charger_modele_ft

CHEMIN_SORTIE = Path("results/ft_transformer")
BIOMARQUEURS = [
    "albumine_g_L", "creatinine_umol_L", "glucose_mmol_L",
    "crp_mg_L", "lymphocytes_pct", "vgm_fL",
    "idr_pct", "pal_U_L", "leucocytes_1000_uL",
]


def charger_test():
    """Charge le test NHANES IV complet (colonnes manquant a 0), applique log1p."""
    test = pd.read_csv("data/processed/test.csv")
    test["crp_mg_L"] = np.log1p(test["crp_mg_L"])
    colonnes = ["age", "est_femme"] + BIOMARQUEURS
    df = test[colonnes].copy()
    for bio in BIOMARQUEURS:
        df[f"manquant_{bio}"] = 0
    y = test["deces_10_ans"].to_numpy()
    return df, y


def predire_avec_ft(df, y):
    """Charge le FT-Transformer, predit sur le test complet, calcule les metriques."""
    modele = charger_modele_ft()
    preds = modele.predict(df)
    colonne_proba = next(
        c for c in preds.columns if c.endswith("_1_probability") or c == "1_probability"
    )
    probas = preds[colonne_proba].to_numpy()
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
    df_test, y_test = charger_test()
    print(f"Test : {len(y_test)} personnes")

    res = predire_avec_ft(df_test, y_test)
    print(f"  AUC={res['auc']:.4f}  Brier={res['brier']:.4f}  ECE={res['ece']:.4f}")

    fig_calib = reliability_diagram({"FT-Transformer": res})
    sauvegarder_figure(fig_calib, CHEMIN_SORTIE / "reliability_diagram.png")

    fig_dist = distribution_probas(res["probas"], res["y_vrai"], "FT-Transformer")
    sauvegarder_figure(fig_dist, CHEMIN_SORTIE / "distribution_probas.png")

    print(f"\nFigures sauvegardees dans {CHEMIN_SORTIE}/")


if __name__ == "__main__":
    main()