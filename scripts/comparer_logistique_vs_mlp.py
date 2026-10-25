"""Compare la regression logistique et le MLP sur le test NHANES IV.

Produit : reliability diagram superpose, courbes ROC, matrices de confusion,
barres comparatives des metriques.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation import ece
from src.mlp import ajouter_masque_complet, charger_modele, predire_probas
from src.visualisation import (
    barres_comparaison_metriques, courbe_roc, distribution_probas, reliability_diagram,
)

CHEMIN_SORTIE = Path("results/comparaison")
BIOMARQUEURS = [
    "albumine_g_L", "creatinine_umol_L", "glucose_mmol_L",
    "crp_mg_L", "lymphocytes_pct", "vgm_fL",
    "idr_pct", "pal_U_L", "leucocytes_1000_uL",
]


def charger_test():
    """Charge le test NHANES IV avec log1p sur la CRP."""
    test = pd.read_csv("data/processed/test.csv")
    test["crp_mg_L"] = np.log1p(test["crp_mg_L"])
    X = test[["age", "est_femme"] + BIOMARQUEURS].values.astype(np.float32)
    y = test["deces_10_ans"].values.astype(np.float32)
    return X, y


def predire_logistique(X, y):
    """Charge et evalue la regression logistique (age + biomarqueurs)."""
    donnees = joblib.load("models/logistique_age_biomarqueurs.pkl")
    # La logistique utilise une entree de 11 colonnes + log(crp), ordre age, est_femme, biomarqueurs
    X_tr = X[:, donnees["colonnes"]]
    X_std = donnees["scaler"].transform(X_tr)
    probas = donnees["modele"].predict_proba(X_std)[:, 1]
    return _metriques(probas, y)


def predire_mlp(X, y):
    """Charge et evalue le MLP."""
    donnees = charger_modele()
    X_std = (X - donnees["moyennes"]) / donnees["ecarts_types"]
    probas = predire_probas(donnees["modele"], ajouter_masque_complet(X_std))
    return _metriques(probas, y)


def _metriques(probas, y_vrai):
    """Fonction pure : renvoie un dict de metriques et de donnees pour les plots."""
    return {
        "probas": probas,
        "y_vrai": y_vrai,
        "auc": float(roc_auc_score(y_vrai, probas)),
        "brier": float(brier_score_loss(y_vrai, probas)),
        "ece": float(ece(y_vrai, probas, n_bins=10)),
    }


def sauvegarder_figure(figure, chemin):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(chemin, dpi=100)


def main():
    X_test, y_test = charger_test()
    resultats = {
        "Logistique": predire_logistique(X_test, y_test),
        "MLP": predire_mlp(X_test, y_test),
    }

    print(f"Test : {len(y_test)} personnes\n")
    for nom, r in resultats.items():
        print(f"  {nom:12s}  AUC={r['auc']:.4f}  Brier={r['brier']:.4f}  ECE={r['ece']:.4f}")

    sauvegarder_figure(reliability_diagram(resultats),
                       CHEMIN_SORTIE / "reliability_diagram.png")
    sauvegarder_figure(courbe_roc(resultats),
                       CHEMIN_SORTIE / "courbes_roc.png")
    sauvegarder_figure(
        barres_comparaison_metriques(
            {nom: {"auc": r["auc"], "brier": r["brier"], "ece": r["ece"]}
             for nom, r in resultats.items()},
            ["auc", "brier", "ece"],
        ),
        CHEMIN_SORTIE / "barres_metriques.png",
    )
    print(f"\nFigures sauvegardees dans {CHEMIN_SORTIE}/")


if __name__ == "__main__":
    main()