import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from tabpfn import TabPFNClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation import ece

CHEMIN_MODELES = Path("models")
CHEMIN_SORTIE = Path("results/tabpfn")
CHEMIN_MODELES.mkdir(exist_ok=True)
CHEMIN_SORTIE.mkdir(parents=True, exist_ok=True)

BIOMARQUEURS = [
    "albumine_g_L", "creatinine_umol_L", "glucose_mmol_L",
    "crp_mg_L", "lymphocytes_pct", "vgm_fL",
    "idr_pct", "pal_U_L", "leucocytes_1000_uL"
]

def charger_donnees():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    
    train["crp_mg_L"] = np.log1p(train["crp_mg_L"])
    test["crp_mg_L"] = np.log1p(test["crp_mg_L"])
    
    colonnes = ["age", "est_femme"] + BIOMARQUEURS
    X_train = train[colonnes].values.astype(np.float32)
    y_train = train["deces_10_ans"].values.astype(np.float32)
    X_test = test[colonnes].values.astype(np.float32)
    y_test = test["deces_10_ans"].values.astype(np.float32)
    
    return X_train, y_train, X_test, y_test

def entrainer_et_evaluer(X_train, y_train, X_test, y_test):
    
    if len(X_train) > 5000:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(X_train), size=5000, replace=False)
        X_train_sub, y_train_sub = X_train[indices], y_train[indices]
    else:
            X_train_sub, y_train_sub = X_train, y_train
        
    modele = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
    modele.fit(X_train_sub, y_train_sub)
    
    probas = modele.predict_proba(X_test)[:, 1]
    
    metriques = {
        "auc": float(roc_auc_score(y_test, probas)),
        "brier": float(brier_score_loss(y_test, probas)),
        "ece": float(ece(y_test, probas, n_bins=10)),
    }
    
    return modele, metriques

def main():
    X_train, y_train, X_test, y_test = charger_donnees()
    print(f"Train complet : {len(y_train)} | Test : {len(y_test)}")
    
    modele, metriques = entrainer_et_evaluer(X_train, y_train, X_test, y_test)
    
    print(f"\nRésultats TabPFN sur le test NHANES IV :")
    print(f"  AUC={metriques['auc']:.4f}  Brier={metriques['brier']:.4f}  ECE={metriques['ece']:.4f}")
    
    joblib.dump(modele, CHEMIN_MODELES / "tabpfn.pkl")
    print(f"\nModèle sauvegardé dans {CHEMIN_MODELES}/tabpfn.pkl")
    
    with open(CHEMIN_SORTIE / "metriques.json", "w") as fichier:
        json.dump(metriques, fichier, indent=2)
    print(f"Métriques sauvegardées dans {CHEMIN_SORTIE}/metriques.json")
    
if __name__ == "__main__":
    main()