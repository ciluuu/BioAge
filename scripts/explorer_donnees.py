"""Genere les figures d'exploration dans results/exploration/."""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CHEMIN_SORTIE = Path("results/exploration")
CHEMIN_SORTIE.mkdir(parents=True, exist_ok=True)

BIOMARQUEURS = [
    "albumine_g_L", "creatinine_umol_L", "glucose_mmol_L",
    "crp_mg_L", "lymphocytes_pct", "vgm_fL",
    "idr_pct", "pal_U_L", "leucocytes_1000_uL",
]


def charger():
    """Charge train et test depuis data/processed/."""
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    print(f"Train : {len(train)} lignes")
    print(f"Test  : {len(test)} lignes")
    return train, test


def figure_distributions_biomarqueurs(train):
    """Histogramme des 9 biomarqueurs sur le train."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, nom in zip(axes.flat, BIOMARQUEURS):
        ax.hist(train[nom], bins=50, color="steelblue", alpha=0.7)
        ax.set_title(nom, fontsize=10)
        ax.set_ylabel("frequence")
    fig.suptitle("Distributions des 9 biomarqueurs (train)")
    fig.tight_layout()
    fig.savefig(CHEMIN_SORTIE / "01_distributions.png", dpi=100)
    plt.close(fig)


def figure_deces_par_age(train):
    """Taux de décès à 10 ans par tranche d'âge de 5 ans."""
    train = train.copy()
    train["tranche_age"] = (train["age"] // 5 * 5).astype(int)
    taux = train.groupby("tranche_age")["deces_10_ans"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(taux.index, taux["mean"] * 100, width=4, color="darkred", alpha=0.7)
    ax.set_xlabel("Tranche d'âge")
    ax.set_ylabel("Taux de décès à 10 ans (%)")
    ax.set_title("Mortalité à 10 ans selon l'âge chronologique (train)")
    fig.tight_layout()
    fig.savefig(CHEMIN_SORTIE / "02_deces_par_age.png", dpi=100)
    plt.close(fig)


def figure_biomarqueurs_par_sexe(train):
    """Comparaison homme/femme pour chaque biomarqueur."""
    train = train.copy()
    train["sexe"] = train["est_femme"].map({0: "Homme", 1: "Femme"})
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, nom in zip(axes.flat, BIOMARQUEURS):
        sns.boxplot(data=train, x="sexe", y=nom, ax=ax, showfliers=False)
        ax.set_title(nom, fontsize=10)
        ax.set_xlabel("")
    fig.suptitle("Biomarqueurs par sexe (train)")
    fig.tight_layout()
    fig.savefig(CHEMIN_SORTIE / "03_biomarqueurs_par_sexe.png", dpi=100)
    plt.close(fig)


def figure_correlations(train):
    """Matrice de corrélation entre biomarqueurs."""
    corr = train[BIOMARQUEURS + ["age"]].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Corrélations entre biomarqueurs et âge (train)")
    fig.tight_layout()
    fig.savefig(CHEMIN_SORTIE / "04_correlations.png", dpi=100)
    plt.close(fig)


def figure_comparaison_train_test(train, test):
    """Compare les distributions train vs test pour chaque biomarqueur."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, nom in zip(axes.flat, BIOMARQUEURS):
        ax.hist(train[nom], bins=50, alpha=0.5, label="train", color="steelblue", density=True)
        ax.hist(test[nom], bins=50, alpha=0.5, label="test", color="darkorange", density=True)
        ax.set_title(nom, fontsize=10)
    axes[0, 0].legend()
    fig.suptitle("Train vs Test : vérifier que les distributions se ressemblent")
    fig.tight_layout()
    fig.savefig(CHEMIN_SORTIE / "05_train_vs_test.png", dpi=100)
    plt.close(fig)


def main():
    train, test = charger()
    figure_distributions_biomarqueurs(train)
    figure_deces_par_age(train)
    figure_biomarqueurs_par_sexe(train)
    figure_correlations(train)
    figure_comparaison_train_test(train, test)
    print(f"\n5 figures ecrites dans {CHEMIN_SORTIE}/")


if __name__ == "__main__":
    main()