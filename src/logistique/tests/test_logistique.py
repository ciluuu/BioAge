"""Test de la prediction logistique."""
from src.logistique import predire_logistique


def test_prediction_adulte_sain():
    """Un adulte 40 ans en bonne sante doit avoir un age biologique proche."""
    age_bio, risque = predire_logistique(
        age=40, albumine=45.0, creatinine=80.0, glucose=5.0,
        crp_mg_L=1.0, lymphocytes_pct=30.0, vgm=90.0, idr=13.0,
        pal=70.0, leucocytes=6.0,
    )
    assert 25 < age_bio < 55
    assert 0 < risque < 0.1