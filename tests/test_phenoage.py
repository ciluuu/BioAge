"""Test de la fonction calculer_phenoage sur un profil connu."""
from src.phenoage import calculer_phenoage


def test_adulte_en_bonne_sante():
    """Un adulte de 40 ans en bonne santé a un âge biologique proche de 40."""
    age_biologique, risque = calculer_phenoage(
        age=40, albumine=45.0, creatinine=80.0, glucose=5.0,
        crp_mg_L=1.0, lymphocytes_pct=30.0, vgm=90.0, idr=13.0,
        pal=70.0, leucocytes=6.0,
    )
    assert 30 < age_biologique < 50
    assert 0 < risque < 0.1