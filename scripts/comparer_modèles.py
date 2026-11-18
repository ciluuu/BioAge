import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from pytorch_tabular import TabularModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation import ece
from src.mlp import ajouter_masque_complet, charger_modele as charger_modele_mlp, predire_probas
from src.visualisation import (
    barres_comparaison_metriques, courbe_roc, reliability_diagram,
)

CHEMIN_SORTIE = Path("results/comparaison_globale")
BIOMARQUEURS = [
    "albumine_g_L", "creatinine_umol_L", "glucose_mmol_L",
    "crp_mg_L", "lymphocytes_pct", "vgm_fL",
    "idr_pct", "pal_U_L", "leucocytes_1000_uL",
]

def charger_donnees_test():
    test = pd.read_csv("data/processed/test.csv")
    test["crp_mg_L"] = np.log1p(test["crp_mg_L"])
    
    colonnes = ["age", "est_femme"] + BIOMARQUEURS
    X = test[colonnes].values.astype(np.float32)
    y = test["deces_10_ans"].values.astype(np.float32)
    return test, X, y

def _metriques(probas, y_vrai):
    return {
        "probas": probas,
        "y_vrai": y_vrai,
        "auc": float(roc_auc_score(y_vrai, probas)),
        "brier": float(brier_score_loss(y_vrai, probas)),
        "ece": float(ece(y_vrai, probas, n_bins=10)),
    }

def predire_logistique(X, y):
    donnees = joblib.load("models/logistique_age_biomarqueurs.pkl")
    X_tr = X[:, donnees["colonnes"]]
    X_std = donnees["scaler"].transform(X_tr)
    probas = donnees["modele"].predict_proba(X_std)[:, 1]
    return _metriques(probas, y)

def predire_mlp(X, y):
    donnees = charger_modele_mlp()
    X_std = (X - donnees["moyennes"]) / donnees["ecarts_types"]
    probas = predire_probas(donnees["modele"], ajouter_masque_complet(X_std))
    return _metriques(probas, y)

def predire_tabpfn(X, y):
    modele = joblib.load("models/tabpfn.pkl")
    probas = modele.predict_proba(X)[:, 1]
    return _metriques(probas, y)

def predire_ft_transformer(df_test, y):
    modele = TabularModel.load_model("models/ft_transformer")
    preds = modele.predict(df_test)
    colonne_proba = next(
    c for c in preds.columns
    if c.endswith("_1_probability") or c == "1_probability"
    )
    probas = preds[colonne_proba].values
    return _metriques(probas, y)

def sauvegarder_figure(figure, chemin):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(chemin, dpi=100)

def main():
    df_test, X_test, y_test = charger_donnees_test()
    print(f"Évaluation sur le test NHANES IV : {len(y_test)} personnes\n")

    resultats = {
        "Logistique": predire_logistique(X_test, y_test),
        "MLP": predire_mlp(X_test, y_test),
        "TabPFN": predire_tabpfn(X_test, y_test),
        "FT-Transformer": predire_ft_transformer(df_test, y_test),
    }

    for nom, r in resultats.items():
        print(f"  {nom:15s}  AUC={r['auc']:.4f}  Brier={r['brier']:.4f}  ECE={r['ece']:.4f}")

    sauvegarder_figure(reliability_diagram(resultats),
                       CHEMIN_SORTIE / "reliability_diagram_4_modeles.png")
    
    sauvegarder_figure(courbe_roc(resultats),
                       CHEMIN_SORTIE / "courbes_roc_4_modeles.png")
    
    sauvegarder_figure(
        barres_comparaison_metriques(
            {nom: {"auc": r["auc"], "brier": r["brier"], "ece": r["ece"]}
             for nom, r in resultats.items()},
            ["auc", "brier", "ece"],
        ),
        CHEMIN_SORTIE / "barres_metriques_4_modeles.png",
    )
    
    print(f"\nFigures de comparaison sauvegardées dans {CHEMIN_SORTIE}/")

if __name__ == "__main__":
    main()