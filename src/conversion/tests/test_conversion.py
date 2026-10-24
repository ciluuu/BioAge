"""Tests pour risque_vers_age_biologique."""
from src.conversion import risque_vers_age_biologique


def test_risque_faible_donne_age_jeune():
    """Un risque de 1% donne un age biologique autour de 35-45 ans."""
    age = risque_vers_age_biologique(0.01)
    assert 30 < age < 50


def test_risque_eleve_donne_age_eleve():
    """Un risque de 50% donne un age biologique autour de 75-85 ans."""
    age = risque_vers_age_biologique(0.5)
    assert 70 < age < 90