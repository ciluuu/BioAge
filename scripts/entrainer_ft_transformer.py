import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
import sys

from pytorch_tabular import TabularModel
from pytorch_tabular.models import FTTransformerConfig
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation import ece

CHEMIN_MODELES = Path("models")
CHEMIN_SORTIE = Path("results/ft_transformer")
CHEMIN_MODELES.mkdir(exist_ok=True)
CHEMIN_SORTIE.mkdir(exist_ok=True)

BIOMARQUEURS = [
    "albumine_g_L", "creatinine_umol_L", "glucose_mmol_L",
    "crp_mg_L", "lymphocytes_pct", "vgm_fL",
    "idr_pct", "pal_U_L", "leucocytes_1000_uL",
]

def charger_et_preparer():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    
    train["crp_mg_L"] = np.log1p(train["crp_mg_L"])
    test["crp_mg_L"] = np.log1p(test["crp_mg_L"])
    
    return train, test

def entrainer_et_evaluer(train_df, test_df):    
    colonnes_continues = ["age"] + BIOMARQUEURS
    colonnes_cat = ["est_femme"]
    
    data_config = DataConfig(
        target=["deces_10_ans"],
        continuous_cols=colonnes_continues,
        categorical_cols=colonnes_cat,
        normalize_continuous_features=True 
    )
    
    model_config = FTTransformerConfig(
        task="classification",
        input_embed_dim=32,
        num_heads=8,
        num_attn_blocks=3,
        learning_rate=1e-3,
    )
    
    trainer_config = TrainerConfig(
        auto_lr_find=True, 
        batch_size=128,
        max_epochs=100, 
        early_stopping_patience=10
    )
    
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=OptimizerConfig(),
        trainer_config=trainer_config
    )
    
    tabular_model.fit(train_df)
    
    preds_df = tabular_model.predict(test_df)
    probas = preds_df["1_probability"].values
    y_test = test_df["deces_10_ans"].values
    
    metriques = {
        "auc": float(roc_auc_score(y_test, probas)),
        "brier": float(brier_score_loss(y_test, probas)),
        "ece": float(ece(y_test, probas, n_bins=10)),
    }
    
    return tabular_model, metriques

def main():
    train_df, test_df = charger_et_preparer()
    print(f"Train : {len(train_df)} | Test : {len(test_df)}")
    
    modele, metriques = entrainer_et_evaluer(train_df, test_df)
    
    print(f"\nRésultats Test NHANES IV :")
    print(f"  AUC={metriques['auc']:.4f}  Brier={metriques['brier']:.4f}  ECE={metriques['ece']:.4f}")
    
    modele.save_model(str(CHEMIN_MODELES / "ft_transformer"))
    
    with open(CHEMIN_SORTIE / "metriques.json", "w") as fichier:
        json.dump(metriques, fichier, indent=2)
        
    print(f"Modèle sauvegardé dans {CHEMIN_MODELES}/ft_transformer/")

if __name__ == "__main__":
    main()