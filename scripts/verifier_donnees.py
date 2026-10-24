"""Vérifie que les .rda NHANES se chargent correctement."""
import pyreadr

for nom in ["NHANES3", "NHANES4"]:
    resultat = pyreadr.read_r(f"data/raw/{nom}.rda")
    df = resultat[nom]
    print(f"\n=== {nom} ===")
    print(f"Dimensions : {df.shape}")
    print(f"Colonnes clés : {[c for c in df.columns if c in ['age', 'gender', 'status', 'permth_int']]}")
    print(f"Âges : {df['age'].min()}-{df['age'].max()}")