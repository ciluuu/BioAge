"""Formule PhenoAge (Levine 2018) : convertit 9 biomarqueurs + l'âge en âge biologique."""
import math


def calculer_phenoage(age, albumine, creatinine, glucose, crp_mg_L,
                      lymphocytes_pct, vgm, idr, pal, leucocytes):
    """Calcule l'âge biologique et le risque de décès à 10 ans.

    Unités attendues :
        age en années, albumine en g/L, creatinine en µmol/L,
        glucose en mmol/L, crp_mg_L en mg/L, lymphocytes_pct en %,
        vgm en fL, idr en %, pal en U/L, leucocytes en 1000 cellules/µL.
    Retourne (age_biologique, risque_10_ans).
    """
    # La formule originale utilise la CRP en mg/dL, on convertit depuis mg/L.
    crp_mg_dL = crp_mg_L / 10.0

    score = (
        -19.907
        - 0.0336 * albumine
        + 0.0095 * creatinine
        + 0.1953 * glucose
        + 0.0954 * math.log(crp_mg_dL)
        - 0.0120 * lymphocytes_pct
        + 0.0268 * vgm
        + 0.3306 * idr
        + 0.00188 * pal
        + 0.0554 * leucocytes
        + 0.0804 * age
    )

    risque_10_ans = 1 - math.exp(-1.51714 * math.exp(score) / 0.0076927)
    age_biologique = 141.50 + math.log(-0.00553 * math.log(1 - risque_10_ans)) / 0.09165

    return age_biologique, risque_10_ans