"""Test de la fonction calculer_phenoage sur un profil connu."""
from src.phenoage import calculer_phenoage


def test_adulte_en_bonne_sante():
    """Vérifie qu'un adulte 40 ans en bonne santé a un âge biologique proche.

    Un profil sain à 40 ans doit renvoyer un âge biologique entre 30 et 50 ans
    et un risque de décès à 10 ans inférieur à 10 %.
    """
    ...
    age_biologique, risque = calculer_phenoage(
        age=40, albumine=45.0, creatinine=80.0, glucose=5.0,
        crp_mg_L=1.0, lymphocytes_pct=30.0, vgm=90.0, idr=13.0,
        pal=70.0, leucocytes=6.0,
    )
    assert 30 < age_biologique < 50
    assert 0 < risque < 0.1

def test_valeurs_extremes_ne_plantent_pas():
    """La formule doit renvoyer un résultat fini même avec des valeurs aberrantes."""
    age_biologique, risque = calculer_phenoage(
        age=19, albumine=33.0, creatinine=33.0, glucose=33.0,
        crp_mg_L=33.0, lymphocytes_pct=33.0, vgm=33.0, idr=33.0,
        pal=33.0, leucocytes=33.0,
    )
    import math
    assert math.isfinite(age_biologique)
    assert 0 < risque < 1