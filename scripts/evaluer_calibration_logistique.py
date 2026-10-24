"""Evalue la calibration des 3 regressions logistiques sur le test.

Produit un reliability diagram et un fichier metriques.json dans results/logistique/.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.evaluation import ece

CHEMIN_MODELES = Path("models")
CHEMIN_SORTIE = Path("results/logistique")
CHEMIN_SORTIE.mkdir(parents=True, exist_ok=True)

BIOMARQUEURS = [
    "albumine_g_L", "creatinine_umol_L", "glucose_mmol_L",
    "crp_mg_L", "lymphocytes_pct", "vgm_fL",
    "idr_pct", "pal_U_L", "leucocytes_1000_uL",
]


def preparer(df):
    """Transforme la CRP en log et renvoie X, y."""
    df = df.copy()
    df["crp_mg_L"] = np.log1p(df["crp_mg_L"])
    X = df[["age", "est_femme"] + BIOMARQUEURS].values
    y = df["deces_10_ans"].values
    return X, y


def evaluer(nom, modele_dict, X, y):
    """Calcule AUC, Brier, ECE pour un modele donne."""
    modele = modele_dict["modele"]
    scaler = modele_dict["scaler"]
    colonnes = modele_dict["colonnes"]
    X_std = scaler.transform(X[:, colonnes])
    probas = modele.predict_proba(X_std)[:, 1]
    return {
        "nom": nom,
        "probas": probas,
        "auc": roc_auc_score(y, probas),
        "brier": brier_score_loss(y, probas),
        "ece": ece(y, probas, n_bins=10),
    }


def tracer_reliability_diagrams(resultats, y_test, chemin):
    """Trace les 3 courbes de calibration superposees + la diagonale ideale."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Calibration parfaite", alpha=0.5)
    for res in resultats:
        proba_moy, taux_reel = calibration_curve(y_test, res["probas"], n_bins=10)
        ax.plot(proba_moy, taux_reel, marker="o",
                label=f"{res['nom']} (ECE={res['ece']:.3f})")
    ax.set_xlabel("Probabilite moyenne predite")
    ax.set_ylabel("Taux de deces reel")
    ax.set_title("Courbes de calibration (test NHANES IV)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(chemin, dpi=100)
    plt.close(fig)


def main():
    test = pd.read_csv("data/processed/test.csv")
    X_test, y_test = preparer(test)
    print(f"Test : {len(y_test)} personnes")

    noms_modeles = ["age_seul", "age_biomarqueurs", "complet"]
    resultats = []
    for nom in noms_modeles:
        modele_dict = joblib.load(CHEMIN_MODELES / f"logistique_{nom}.pkl")
        res = evaluer(nom, modele_dict, X_test, y_test)
        print(f"  {nom:20s}  AUC={res['auc']:.4f}  Brier={res['brier']:.4f}  ECE={res['ece']:.4f}")
        resultats.append(res)

    tracer_reliability_diagrams(resultats, y_test, CHEMIN_SORTIE / "reliability_diagram.png")

    # Sauvegarde des metriques (sans les probas, pour avoir un JSON leger)
    metriques = [{k: v for k, v in r.items() if k != "probas"} for r in resultats]
    with open(CHEMIN_SORTIE / "metriques.json", "w") as fichier:
        json.dump(metriques, fichier, indent=2)
    print(f"\nMetriques et reliability diagram sauvegardes dans {CHEMIN_SORTIE}/")


if __name__ == "__main__":
    main()