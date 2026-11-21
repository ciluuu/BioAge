"""Cross-validation 5-folds pour le FT-Transformer (avec masquage des manquants).

Chaque fold entraine sur un jeu augmente (version complete + copies masquees,
colonnes manquant_<X>) et evalue sur le fold de validation COMPLET.
Memes hyperparametres que scripts/entrainer_ft_transformer.py.
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
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.evaluation import ece
from src.masquage.masquage import tirer_masque

CHEMIN_TRAIN = Path("data/processed/train.csv")
CHEMIN_SORTIE = Path("results/ft_transformer/cv_5_folds.json")
NOMBRE_FOLDS = 5
GRAINE = 42
BIOMARQUEURS = [
    "albumine_g_L", "creatinine_umol_L", "glucose_mmol_L",
    "crp_mg_L", "lymphocytes_pct", "vgm_fL",
    "idr_pct", "pal_U_L", "leucocytes_1000_uL",
]
COLONNES_CONTINUES = ["age"] + BIOMARQUEURS
COLONNES_CAT = ["est_femme"] + [f"manquant_{bio}" for bio in BIOMARQUEURS]


def charger_donnees():
    """Charge le train, applique log1p sur la CRP."""
    train = pd.read_csv(CHEMIN_TRAIN)
    train["crp_mg_L"] = np.log1p(train["crp_mg_L"])
    colonnes = ["age", "est_femme"] + BIOMARQUEURS + ["deces_10_ans"]
    return train[colonnes].copy()


def construire_augmente(df, moyennes_bio, n_copies=3, taux_max=7, graine=GRAINE):
    """Version complete + n_copies masquees, avec colonnes manquant_<X>."""
    rng = np.random.default_rng(graine)
    parties = []

    complet = df.copy()
    for bio in BIOMARQUEURS:
        complet[f"manquant_{bio}"] = 0
    parties.append(complet)

    for _ in range(n_copies):
        masque = tirer_masque(len(df), taux_max=taux_max, graine=int(rng.integers(1_000_000)))
        masque_bio = masque[:, 2:]
        copie = df.copy()
        for j, bio in enumerate(BIOMARQUEURS):
            est_manquant = masque_bio[:, j] == 0
            copie.loc[est_manquant, bio] = moyennes_bio[bio]
            copie[f"manquant_{bio}"] = est_manquant.astype(int)
        parties.append(copie)

    return pd.concat(parties, ignore_index=True)


def ajouter_colonnes_manquant_a_zero(df):
    """Colonnes manquant_<X> toutes a 0 (fold de validation complet)."""
    df = df.copy()
    for bio in BIOMARQUEURS:
        df[f"manquant_{bio}"] = 0
    return df


def construire_modele():
    """FT-Transformer, memes hyperparametres que l'entrainement."""
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


def faire_cv():
    """CV 5-folds : entrainement augmente, evaluation sur fold complet."""
    df = charger_donnees()
    cible = df["deces_10_ans"]
    kfold = StratifiedKFold(n_splits=NOMBRE_FOLDS, shuffle=True, random_state=GRAINE)

    aucs, briers, eces = [], [], []

    for num_fold, (idx_tr, idx_val) in enumerate(kfold.split(df, cible), start=1):
        train_fold = df.iloc[idx_tr].reset_index(drop=True)
        val_fold = df.iloc[idx_val].reset_index(drop=True)

        moyennes_bio = train_fold[BIOMARQUEURS].mean()
        train_aug = construire_augmente(train_fold, moyennes_bio)

        modele = construire_modele()
        modele.fit(train=train_aug)

        val_complet = ajouter_colonnes_manquant_a_zero(val_fold)
        preds = modele.predict(val_complet.drop(columns=["deces_10_ans"]))
        colonne_proba = next(
            c for c in preds.columns if c.endswith("_1_probability") or c == "1_probability"
        )
        probas = preds[colonne_proba].to_numpy()
        y_val = val_fold["deces_10_ans"].to_numpy()

        auc = roc_auc_score(y_val, probas)
        brier = brier_score_loss(y_val, probas)
        ece_val = ece(y_val, probas, n_bins=10)

        aucs.append(auc)
        briers.append(brier)
        eces.append(ece_val)
        print(f"  Fold {num_fold}/5 : AUC={auc:.4f}  Brier={brier:.4f}  ECE={ece_val:.4f}")

    return {
        "auc": {"valeurs": aucs, "moyenne": float(np.mean(aucs)), "ecart_type": float(np.std(aucs))},
        "brier": {"valeurs": briers, "moyenne": float(np.mean(briers)), "ecart_type": float(np.std(briers))},
        "ece": {"valeurs": eces, "moyenne": float(np.mean(eces)), "ecart_type": float(np.std(eces))},
    }


def main():
    print("Cross-validation 5-folds - FT-Transformer (avec masquage)")
    print("-" * 60)
    resultats = faire_cv()
    print("-" * 60)
    print(f"AUC   : {resultats['auc']['moyenne']:.4f} +/- {resultats['auc']['ecart_type']:.4f}")
    print(f"Brier : {resultats['brier']['moyenne']:.4f} +/- {resultats['brier']['ecart_type']:.4f}")
    print(f"ECE   : {resultats['ece']['moyenne']:.4f} +/- {resultats['ece']['ecart_type']:.4f}")

    CHEMIN_SORTIE.parent.mkdir(parents=True, exist_ok=True)
    with CHEMIN_SORTIE.open("w") as f:
        json.dump(resultats, f, indent=2)
    print(f"\nSauvegarde : {CHEMIN_SORTIE}")


if __name__ == "__main__":
    main()