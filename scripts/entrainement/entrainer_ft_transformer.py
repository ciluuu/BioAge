"""Entraine le FT-Transformer avec gestion explicite des biomarqueurs manquants.

Chaque personne apparait en version complete + 3 versions ou des biomarqueurs
sont masques (remplaces par la moyenne du train). Une colonne 'manquant_<X>'
indique au modele lesquels sont absents : il apprend ainsi un embedding pour
'biomarqueur manquant', comme suggere par l'enonce.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import FTTransformerConfig
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.evaluation import ece
from src.masquage.masquage import tirer_masque

CHEMIN_MODELES = Path("models")
CHEMIN_SORTIE = Path("results/ft_transformer")
CHEMIN_MODELES.mkdir(exist_ok=True)
CHEMIN_SORTIE.mkdir(parents=True, exist_ok=True)

GRAINE = 42
BIOMARQUEURS = [
    "albumine_g_L", "creatinine_umol_L", "glucose_mmol_L",
    "crp_mg_L", "lymphocytes_pct", "vgm_fL",
    "idr_pct", "pal_U_L", "leucocytes_1000_uL",
]
COLONNES_CONTINUES = ["age"] + BIOMARQUEURS
COLONNES_CAT = ["est_femme"] + [f"manquant_{bio}" for bio in BIOMARQUEURS]


def charger_donnees():
    """Charge train/test, applique log1p sur la CRP."""
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    train["crp_mg_L"] = np.log1p(train["crp_mg_L"])
    test["crp_mg_L"] = np.log1p(test["crp_mg_L"])
    colonnes = ["age", "est_femme"] + BIOMARQUEURS + ["deces_10_ans"]
    return train[colonnes].copy(), test[colonnes].copy()


def construire_augmente(df, moyennes_bio, n_copies=3, taux_max=7, graine=GRAINE):
    """Version complete + n_copies masquees, avec colonnes manquant_<X> (0/1).

    Les biomarqueurs masques prennent la moyenne du train ; manquant_<X>=1
    indique au modele qu'ils sont absents.
    """
    rng = np.random.default_rng(graine)
    parties = []

    complet = df.copy()
    for bio in BIOMARQUEURS:
        complet[f"manquant_{bio}"] = 0
    parties.append(complet)

    for _ in range(n_copies):
        masque = tirer_masque(len(df), taux_max=taux_max, graine=int(rng.integers(1_000_000)))
        masque_bio = masque[:, 2:]                     # (n, 9) : 1 present / 0 manquant
        copie = df.copy()
        for j, bio in enumerate(BIOMARQUEURS):
            est_manquant = masque_bio[:, j] == 0
            copie.loc[est_manquant, bio] = moyennes_bio[bio]
            copie[f"manquant_{bio}"] = est_manquant.astype(int)
        parties.append(copie)

    return pd.concat(parties, ignore_index=True)


def ajouter_colonnes_manquant_a_zero(df):
    """Ajoute les colonnes manquant_<X> toutes a 0 (donnees completes)."""
    df = df.copy()
    for bio in BIOMARQUEURS:
        df[f"manquant_{bio}"] = 0
    return df


def construire_modele():
    """FT-Transformer : memes hyperparametres qu'avant + colonnes manquant en categoriel."""
    return TabularModel(
        data_config=DataConfig(
            target=["deces_10_ans"],
            continuous_cols=COLONNES_CONTINUES,
            categorical_cols=COLONNES_CAT,
            normalize_continuous_features=True,
        ),
        model_config=FTTransformerConfig(
            task="classification",
            input_embed_dim=32,
            num_heads=8,
            num_attn_blocks=3,
            learning_rate=1e-3,
        ),
        optimizer_config=OptimizerConfig(),
        trainer_config=TrainerConfig(
            auto_lr_find=False,
            batch_size=128,
            max_epochs=100,
            early_stopping_patience=10,
        ),
    )


def main():
    train, test = charger_donnees()
    print(f"Train : {len(train)} · Test : {len(test)}")

    moyennes_bio = train[BIOMARQUEURS].mean()
    train_aug = construire_augmente(train, moyennes_bio)
    print(f"Train augmente : {len(train_aug)} lignes")

    modele = construire_modele()
    modele.fit(train=train_aug)

    # Evaluation sur le test COMPLET (colonnes manquant a 0)
    test_complet = ajouter_colonnes_manquant_a_zero(test)
    preds = modele.predict(test_complet.drop(columns=["deces_10_ans"]))
    colonne_proba = next(
        c for c in preds.columns if c.endswith("_1_probability") or c == "1_probability"
    )
    probas = preds[colonne_proba].to_numpy()
    y_test = test["deces_10_ans"].to_numpy()
    metriques = {
        "auc": float(roc_auc_score(y_test, probas)),
        "brier": float(brier_score_loss(y_test, probas)),
        "ece": float(ece(y_test, probas, n_bins=10)),
    }
    print(f"  AUC={metriques['auc']:.4f}  Brier={metriques['brier']:.4f}  ECE={metriques['ece']:.4f}")

    modele.save_model(str(CHEMIN_MODELES / "ft_transformer"))
    with open(CHEMIN_SORTIE / "metriques.json", "w") as fichier:
        json.dump(metriques, fichier, indent=2)
    print(f"\nModele dans {CHEMIN_MODELES}/ft_transformer/ · metriques dans {CHEMIN_SORTIE}/metriques.json")


if __name__ == "__main__":
    main()