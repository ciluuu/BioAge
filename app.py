import streamlit as st
import math

with st.form("biomarqueurs"):
    age = st.number_input("Age", min_value=0, max_value=120, value=None)
    col1, col2 = st.columns(2)
    with col1:
        albumine = st.number_input("Albumine (g/L)", min_value=0.0, value=None, placeholder="ex : 35.0")
        creatinine = st.number_input("Créatinine (µmol/L)", min_value=0.0, value=None, placeholder="ex : 80.0")
        glucose = st.number_input("Glucose (mmol/L)", min_value=0.0, value=None, placeholder="ex : 5.5")
        crp = st.number_input("Protéine C-reactive (CRP) (mg/L)", min_value=0.0, value=None, placeholder="ex : 5.0")
        lymphocytes = st.number_input("Lymphocytes (%)", min_value=0.0, value=None, placeholder="ex : 1.5")
    with col2:
        vgm = st.number_input("Volume Globulaire Moyen (VGM) (fL)", min_value=0.0, value=None, placeholder="ex : 90.0")
        idr = st.number_input("Indice de Distribution des globules Rouges (IDR) (%)", min_value=0.0, value=None, placeholder="ex : 15.0")
        pal = st.number_input("Phosphatases alcalines (PAL) (U/L)", min_value=0.0, value=None, placeholder="ex : 100.0")
        leucocytes = st.number_input("Leucocytes (1000 cells/uL)", min_value=0.0, value=None, placeholder="ex : 7.0")
    submit_button = st.form_submit_button("Envoyer", type="primary", use_container_width=True)

if submit_button:
    if age is None:
        st.error("Veuillez renseigner votre âge.")
    else:
        biomarqueurs = {
            "Albumine": albumine,
            "Créatinine": creatinine,
            "Glucose": glucose,
            "Protéine C-reactive (CRP)": crp,
            "Lymphocytes": lymphocytes,
            "Volume Globulaire Moyen (VGM)": vgm,
            "Indice de Distribution des globules Rouges (IDR)": idr,
            "Phosphatases alcalines (PAL)": pal,
            "Leucocytes": leucocytes
        }
        for nom, valeur in biomarqueurs.items():
            if valeur is None:
                st.error(f"Veuillez renseigner la valeur pour {nom}.")
                break
            
        st.success("Formulaire envoyé avec succès")
        
def calculer_phenoage (age, albumine, creatinine, glucose, crp, lymphocytes, vgm, idr, pal, leucocytes):
    # coefficients pour chaque biomarqueur
    coefficients = {
        albumine: -0.0336,
        creatinine: 0.0095,
        glucose: 0.1953,
        crp: 0.0954,
        lymphocytes: -0.0120,
        vgm: 0.0268,
        idr: 0.3306,
        pal: 0.0019,
        leucocytes: 0.0554,
        age: 0.0804
    }
    xb = (-19.907 + 
        coefficients[albumine] * albumine +
        coefficients[creatinine] * creatinine +
        coefficients[glucose] * glucose +
        coefficients[crp] * math.log(crp) +
        coefficients[lymphocytes] * lymphocytes +
        coefficients[vgm] * vgm +
        coefficients[idr] * idr +
        coefficients[pal] * pal +
        coefficients[leucocytes] * leucocytes +
        coefficients[age] * age)
    M = 1 - math.exp((-1.51714 * math.exp(xb)) / 0.0076927)
    phenoage = 141.5 + (math.log(-0.00553 * math.log(1 - M)) / 0.09165)
    return phenoage, M

def tester_phenoage():
    valeurs_test = {
        "age": 40,
        "albumine": 35.0,
        "creatinine": 80.0,
        "glucose": 5.5,
        "crp": 5.0,
        "lymphocytes": 1.5,
        "vgm": 90.0,
        "idr": 15.0,
        "pal": 100.0,
        "leucocytes": 7.0
    }
    phenoage, M = calculer_phenoage(**valeurs_test)
    print(f"Test PhenoAge avec valeurs : {valeurs_test}")
    print(f"PhenoAge: {phenoage}, M: {M}")

tester_phenoage()