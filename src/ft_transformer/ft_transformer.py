from pathlib import Path
import numpy as np
import pandas as pd
from pytorch_tabular import TabularModel

from src.conversion.conversion import risque_vers_age_biologique

CHEMIN_MODELE = Path("models/ft_transformer")

def charger_modele_ft(chemin=CHEMIN_MODELE):
    return TabularModel.load_model(str(chemin))

def predire_pour_une_personne_ft(modele, valeurs_11):
    colonnes = [
        "age", "est_femme", "albumine_g_L", "creatinine_umol_L", 
        "glucose_mmol_L", "crp_mg_L", "lymphocytes_pct", 
        "vgm_fL", "idr_pct", "pal_U_L", "leucocytes_1000_uL"
    ]
    
    df_entree = pd.DataFrame([valeurs_11], columns=colonnes)
    
    preds_df = modele.predict(df_entree)
    risque = float(preds_df["1_probability"].iloc[0])
    
    return risque_vers_age_biologique(risque), risque