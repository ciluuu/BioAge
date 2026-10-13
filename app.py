import streamlit as st

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
        renseignes = {nom: valeur for nom, valeur in biomarqueurs.items() if valeur is not None}
        for nom, valeur in renseignes.items():
            if valeur is None:
                st.error(f"Veuillez renseigner la valeur pour {nom}.")
                break
            
        st.success("Formulaire envoyé avec succès")