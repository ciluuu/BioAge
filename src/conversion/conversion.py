"""Conversion probabilite de deces a 10 ans vers age biologique.

Formule d'inversion de Gompertz utilisee dans PhenoAge (Levine 2018).
Applicable a n'importe quel modele qui predit un risque dans [0, 1].
"""
import math


def risque_vers_age_biologique(risque_10_ans):
    """Convertit une probabilite de deces a 10 ans en age biologique.

    Parameters
    ----------
    risque_10_ans : float
        Probabilite de deces dans les 10 ans, dans [0, 1].

    Returns
    -------
    float
        Age biologique estime en annees.
    """
    EPSILON = 1e-10
    risque = max(EPSILON, min(1 - EPSILON, risque_10_ans))
    return 141.50 + math.log(-0.00553 * math.log(1 - risque)) / 0.09165