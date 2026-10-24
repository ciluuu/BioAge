"""Prépare train.csv (NHANES III) et test.csv (NHANES IV vagues 1-3).

Pour chaque personne : 9 biomarqueurs + age + sexe + deces_10_ans.
"""
import pyreadr
import pandas as pd

# --- Correspondances entre colonnes BioAge et nos noms clairs ---
RENOMMAGE = {
    "albumin_gL":    "albumine_g_L",
    "creat_umol":    "creatinine_umol_L",
    "glucose_mmol":  "glucose_mmol_L",
    "crp":           "crp_mg_dL",         # NHANES stocke en mg/dL (convention US)
    "lymph":         "lymphocytes_pct",
    "mcv":           "vgm_fL",
    "rdw":           "idr_pct",
    "alp":           "pal_U_L",
    "wbc":           "leucocytes_1000_uL",
}

BIOMARQUEURS = list(RENOMMAGE.values())


def preparer(df, nom):
    """Nettoie un dataset NHANES et renvoie un DataFrame prêt pour le ML."""
    print(f"\n=== {nom} ===")
    print(f"Lignes initiales : {len(df)}")

    # Renommage
    df = df.rename(columns=RENOMMAGE)

    # Création de la cible : décès à 10 ans
    df = df[df["eligstat"] == 1].copy()        # garder uniquement les éligibles au suivi
    cond_decede = (df["status"] == 1) & (df["permth_int"] < 120)
    cond_vivant = (df["status"] == 0) & (df["permth_int"] >= 120)
    df["deces_10_ans"] = cond_decede.astype(int)
    df = df[cond_decede | cond_vivant]         # exclure les cas incertains

    # Encodage du sexe (0 = homme, 1 = femme)
    df["est_femme"] = (df["gender"] == 2).astype(int)

    # Garder uniquement les colonnes utiles + conversion CRP mg/dL -> mg/L
    df["crp_mg_L"] = df["crp_mg_dL"] * 10.0
    colonnes_finales = ["sampleID", "age", "est_femme",
                        "albumine_g_L", "creatinine_umol_L", "glucose_mmol_L",
                        "crp_mg_L", "lymphocytes_pct", "vgm_fL",
                        "idr_pct", "pal_U_L", "leucocytes_1000_uL",
                        "deces_10_ans"]
    df = df[colonnes_finales]

    # Cas complets uniquement
    avant = len(df)
    df = df.dropna()
    print(f"Après filtrage et dropna : {len(df)} ({avant - len(df)} lignes avec NA supprimées)")
    print(f"Décès à 10 ans : {df['deces_10_ans'].sum()} ({df['deces_10_ans'].mean():.1%})")
    print(f"Femmes : {df['est_femme'].mean():.1%}")

    return df


def main():
    # Charger
    n3 = pyreadr.read_r("data/raw/NHANES3.rda")["NHANES3"]
    n4 = pyreadr.read_r("data/raw/NHANES4.rda")["NHANES4"]

    # NHANES IV : garder uniquement les vagues 1 à 3 (10 ans de recul garantis)
    print(f"\nNHANES4 : vagues disponibles = {sorted(n4['wave'].unique())}")
    n4 = n4[n4["wave"].isin([1, 2, 3])]

    # Préparer et sauvegarder
    train = preparer(n3, "NHANES III (train)")
    test = preparer(n4, "NHANES IV vagues 1-3 (test)")

    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    print("\n✅ Fichiers sauvegardés dans data/processed/")


if __name__ == "__main__":
    main()