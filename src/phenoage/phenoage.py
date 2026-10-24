"""Formule PhenoAge (Levine 2018) : convertit 9 biomarqueurs + l'âge en âge biologique."""
import math
from src.conversion import risque_vers_age_biologique

def calculer_phenoage(age, albumine, creatinine, glucose, crp_mg_L,
                      lymphocytes_pct, vgm, idr, pal, leucocytes):
    """Calcule l'âge biologique PhenoAge (Levine 2018).

    Parameters
    ----------
    age : float
        Âge en années.
    albumine, creatinine, glucose : float
        g/L, µmol/L, mmol/L.
    crp_mg_L : float
        CRP en mg/L (convertie en interne en mg/dL).
    lymphocytes_pct : float
        Lymphocytes en %.
    vgm, idr, pal, leucocytes : float
        fL, %, U/L, 1000 cellules/µL.

    Returns
    -------
    age_biologique : float
        Âge biologique en années.
    risque_10_ans : float
        Probabilité de décès à 10 ans, dans [0, 1].
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
    # Clamp pour cohérence du contrat (risque strictement dans ]0, 1[)
    EPSILON = 1e-10
    risque_10_ans = max(EPSILON, min(1 - EPSILON, risque_10_ans))
    age_biologique = risque_vers_age_biologique(risque_10_ans)
    return age_biologique, risque_10_ans
