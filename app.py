"""Application Streamlit : formulaire de 9 biomarqueurs, affichage de l'âge biologique."""
import streamlit as st

from src.phenoage import calculer_phenoage
from src.biomarqueurs import BIOMARQUEURS


st.title("BioAge — Estimation de l'âge biologique")
st.caption("Outil de prévention, pas un diagnostic médical.")

# Formulaire
with st.form("formulaire"):
    age = st.number_input("Âge (années)", min_value=0, max_value=120, value=None)

    col_gauche, col_droite = st.columns(2)
    saisies = {}
    for index, bio in enumerate(BIOMARQUEURS):
        colonne = col_gauche if index % 2 == 0 else col_droite
        saisies[bio["cle"]] = colonne.number_input(
            f"{bio['label']} ({bio['unite']})",
            min_value=float(bio["min"]),
            max_value=float(bio["max"]),
            value=None,
            placeholder=f"ex : {bio['exemple']}",
        )

    envoye = st.form_submit_button(
        "Calculer mon âge biologique", type="primary", use_container_width=True
    )

# Résultat
if envoye:
    manquants = [bio["label"] for bio in BIOMARQUEURS if saisies[bio["cle"]] is None]

    if age is None:
        st.error("Veuillez renseigner votre âge.")
    elif manquants:
        st.error("Valeurs manquantes : " + ", ".join(manquants))
    else:
        age_biologique, risque = calculer_phenoage(
            age=age,
            albumine=saisies["albumine"],
            creatinine=saisies["creatinine"],
            glucose=saisies["glucose"],
            crp_mg_L=saisies["crp"],
            lymphocytes_pct=saisies["lymphocytes"],
            vgm=saisies["vgm"],
            idr=saisies["idr"],
            pal=saisies["pal"],
            leucocytes=saisies["leucocytes"],
        )

        colonne_1, colonne_2, colonne_3 = st.columns(3)
        colonne_1.metric("Âge biologique", f"{age_biologique:.1f} ans")
        colonne_2.metric("Écart avec l'âge réel", f"{age_biologique - age:+.1f} ans")
        colonne_3.metric("Risque de décès à 10 ans", f"{risque * 100:.1f} %")