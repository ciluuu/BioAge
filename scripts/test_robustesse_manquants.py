"""Test de robustesse aux biomarqueurs manquants : compare MLP et logistique.

Pour chaque niveau de biomarqueurs caches, calcule l'AUC des deux modeles sur le test.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.mlp import ajouter_masque_complet, charger_modele, predire_probas

CHEMIN_SORTIE = Path("results/comparaison")
GRAINE = 42
BIOMARQUEURS = [
    "albumine_g_L", "creatinine_umol_L", "glucose_mmol_L",
    "crp_mg_L", "lymphocytes_pct", "vgm_fL",
    "idr_pct", "pal_U_L", "leucocytes_1000_uL",
]


def charger_test():
    """Charge le test et prepare X, y."""
    test = pd.read_csv("data/processed/test.csv")
    test["crp_mg_L"] = np.log1p(test["crp_mg_L"])
    X = test[["age", "est_femme"] + BIOMARQUEURS].values.astype(np.float32)
    y = test["deces_10_ans"].values.astype(np.float32)
    return X, y


def generer_masques(nb_lignes, nb_a_cacher, graine):
    """Pour chaque ligne, choisit nb_a_cacher biomarqueurs a cacher (parmi les 9)."""
    rng = np.random.default_rng(graine)
    masques = np.ones((nb_lignes, 11), dtype=np.float32)
    for i in range(nb_lignes):
        if nb_a_cacher > 0:
            idx = rng.choice(range(2, 11), size=nb_a_cacher, replace=False)
            masques[i, idx] = 0.0
    return masques


def auc_mlp(donnees_mlp, X, y, masques):
    """AUC du MLP avec les biomarqueurs caches via le masque natif."""
    X_std = (X - donnees_mlp["moyennes"]) / donnees_mlp["ecarts_types"]
    X_std = X_std * masques
    entree = np.concatenate([X_std, masques], axis=1).astype(np.float32)
    probas = predire_probas(donnees_mlp["modele"], entree)
    return float(roc_auc_score(y, probas))


def auc_logistique(donnees_log, X, y, masques, moyennes_train):
    """AUC de la logistique en imputant les biomarqueurs caches a la moyenne du train."""
    X_impute = np.where(masques == 1, X, moyennes_train)
    X_tr = X_impute[:, donnees_log["colonnes"]]
    X_std = donnees_log["scaler"].transform(X_tr)
    probas = donnees_log["modele"].predict_proba(X_std)[:, 1]
    return float(roc_auc_score(y, probas))


def tracer_courbes(resultats, chemin):
    """Trace l'AUC en fonction du nombre de biomarqueurs caches."""
    niveaux = sorted(resultats.keys())
    aucs_mlp = [resultats[n]["mlp"] for n in niveaux]
    aucs_log = [resultats[n]["logistique"] for n in niveaux]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(niveaux, aucs_mlp, marker="o", label="MLP (masque natif)", linewidth=2)
    ax.plot(niveaux, aucs_log, marker="s", label="Logistique (imputation moyenne)", linewidth=2)
    ax.set_xlabel("Nombre de biomarqueurs caches")
    ax.set_ylabel("AUC sur le test NHANES IV")
    ax.set_title("Robustesse aux biomarqueurs manquants")
    ax.set_xticks(niveaux)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chemin, dpi=100)


def main():
    X_test, y_test = charger_test()
    donnees_mlp = charger_modele()
    donnees_log = joblib.load("models/logistique_age_biomarqueurs.pkl")

    # Moyennes du train, pour l'imputation (meme stat qu'a l'entrainement de la logistique)
    train = pd.read_csv("data/processed/train.csv")
    train["crp_mg_L"] = np.log1p(train["crp_mg_L"])
    moyennes_train = train[["age", "est_femme"] + BIOMARQUEURS].mean().values

    resultats = {}
    for nb_caches in range(0, 9):
        masques = generer_masques(len(X_test), nb_caches, graine=GRAINE)
        auc_m = auc_mlp(donnees_mlp, X_test, y_test, masques)
        auc_l = auc_logistique(donnees_log, X_test, y_test, masques, moyennes_train)
        resultats[nb_caches] = {"mlp": auc_m, "logistique": auc_l}
        print(f"  {nb_caches} caches : MLP AUC={auc_m:.4f}  |  Logistique AUC={auc_l:.4f}")

    CHEMIN_SORTIE.mkdir(parents=True, exist_ok=True)
    tracer_courbes(resultats, CHEMIN_SORTIE / "robustesse_manquants.png")
    print(f"\nFigure sauvegardee dans {CHEMIN_SORTIE}/robustesse_manquants.png")


if __name__ == "__main__":
    main()