"""
Agrège les résultats des 5-folds CV des 4 modèles et produit tableau + figure.

Lit results/<modele>/cv_5_folds.json pour chaque modèle, écrit :
- results/comparaison_globale/tableau_cv.csv : pour le rapport Word
- results/comparaison_globale/figure_cv.png  : bar chart groupé avec barres d'erreur

À lancer après les 4 scripts cv_*.py.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHEMIN_SORTIE = Path("results/comparaison_globale")
METRIQUES = ("auc", "brier", "ece")
MODELES = {
    "Logistique":     "results/logistique/cv_5_folds.json",
    "MLP":            "results/mlp/cv_5_folds.json",
    "TabPFN":         "results/tabpfn/cv_5_folds.json",
    "FT-Transformer": "results/ft_transformer/cv_5_folds.json",
}


def charger_resume(chemin):
    """Lit un JSON de CV et renvoie le dict contenant auc, brier, ece."""
    with open(chemin) as fichier:
        return json.load(fichier)


def construire_tableau():
    """
    Lit les 4 JSON et construit un DataFrame long format.
    Colonnes : modele, metrique, moyenne, ecart_type.
    """
    lignes = []
    for nom_modele, chemin in MODELES.items():
        resume = charger_resume(chemin)
        for metrique in METRIQUES:
            lignes.append({
                "modele": nom_modele,
                "metrique": metrique,
                "moyenne": resume[metrique]["moyenne"],
                "ecart_type": resume[metrique]["ecart_type"],
            })
    return pd.DataFrame(lignes)


def tracer_figure(donnees, chemin):
    """3 subplots cote a cote, un par metrique, avec echelle adaptee a chaque."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    noms = list(MODELES.keys())
    positions = np.arange(len(noms))
    couleurs = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax, metrique in zip(axes, METRIQUES):
        sous = donnees[donnees["metrique"] == metrique].set_index("modele").loc[noms]
        ax.bar(positions, sous["moyenne"], yerr=sous["ecart_type"],
               color=couleurs, capsize=5, edgecolor="black", linewidth=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(noms, rotation=15, ha="right")
        ax.set_title(metrique.upper())
        ax.grid(axis="y", alpha=0.3)
        # Zoom automatique autour des valeurs
        marge = sous["ecart_type"].max() * 3
        ax.set_ylim(sous["moyenne"].min() - marge, sous["moyenne"].max() + marge)

    fig.suptitle("Cross-validation 5-folds - moyenne +/- ecart-type")
    plt.tight_layout()
    plt.savefig(chemin, dpi=150)
    plt.close()


def main():
    print("Agregation des 5-folds CV sur les 4 modeles")
    print("-" * 60)

    CHEMIN_SORTIE.mkdir(parents=True, exist_ok=True)
    donnees = construire_tableau()

    tableau = donnees.pivot(index="modele", columns="metrique",
                            values=["moyenne", "ecart_type"])
    print(tableau.round(4))

    donnees.to_csv(CHEMIN_SORTIE / "tableau_cv.csv", index=False)
    tracer_figure(donnees, CHEMIN_SORTIE / "figure_cv.png")
    print(f"\nTableau et figure sauvegardes dans {CHEMIN_SORTIE}")


if __name__ == "__main__":
    main()