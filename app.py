"""Application Streamlit : formulaire de 9 biomarqueurs, affichage de l'âge biologique."""
import streamlit as st

from src.biomarqueurs import BIOMARQUEURS
from src.phenoage import calculer_phenoage
from src.logistique import predire_logistique
from src.mlp import charger_modele as charger_modele_mlp
from src.mlp import predire_pour_une_personne as predire_mlp_personne


st.title("BioAge - Estimation de l'âge biologique")
st.caption("Outil de prévention, pas un diagnostic médical.")

modele_choisi = st.radio(
    "Modele d'estimation",
    options=[
        "PhenoAge (formule)",
        "Regression logistique (ML)",
        "MLP (deep learning)",
    ],
    horizontal=True,
    help="PhenoAge : formule de reference. "
         "Logistique : baseline ML. "
         "MLP : deep learning, meilleure calibration.",
)
# Formulaire
with st.form("formulaire"):
    age = st.number_input("Âge (années)", min_value=0, max_value=120, value=None)

    sexe = st.radio(
        "Sexe (optionnel)",
        options=["Non précisé", "Femme", "Homme"],
        horizontal=True,
        help="Utilise par certains modeles. Si non precise, une valeur moyenne est utilisee.",
    )

    col_gauche, col_droite = st.columns(2)
    saisies = {}
    for index, bio in enumerate(BIOMARQUEURS):
        colonne = col_gauche if index % 2 == 0 else col_droite

        aides = []
        if bio.get("plage_min") is not None and bio.get("plage_max") is not None:
            aides.append(
                f"Plage normale : {bio['plage_min']} - {bio['plage_max']} {bio['unite']}"
            )
        if bio.get("plage_min_femme") is not None and bio.get("plage_max_femme") is not None:
            aides.append(
                f"Plage normale pour les femmes : "
                f"{bio['plage_min_femme']} - {bio['plage_max_femme']} {bio['unite']}"
            )
        if bio.get("plage_min_homme") is not None and bio.get("plage_max_homme") is not None:
            aides.append(
                f"Plage normale pour les hommes : "
                f"{bio['plage_min_homme']} - {bio['plage_max_homme']} {bio['unite']}"
            )
        help_text = "\n".join(aides)

        saisies[bio["cle"]] = colonne.number_input(
            f"{bio['label']} ({bio['unite']})",
            min_value=float(bio["min"]),
            value=None,
            help=help_text if help_text else None,
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
    elif manquants and not modele_choisi.startswith("MLP"):
        st.error("Ce modele ne gere pas les biomarqueurs manquants : " + ", ".join(manquants))
        st.info("Le MLP, lui, accepte les valeurs manquantes. Essayez-le.")
    else:
        if modele_choisi.startswith("PhenoAge"):
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
        elif modele_choisi.startswith("Regression"):
            age_biologique, risque = predire_logistique(
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
        else:  # MLP (avec gestion des manquants)
            import numpy as np
            from src.mlp import predire_avec_manquants

            # Encodage du sexe : Femme=1, Homme=0, Non precise=0.5 (valeur moyenne)
            if sexe == "Femme":
                est_femme = 1.0
            elif sexe == "Homme":
                est_femme = 0.0
            else:
                est_femme = 0.5

            donnees_mlp = charger_modele_mlp()
            cles = ["albumine", "creatinine", "glucose", "crp",
                    "lymphocytes", "vgm", "idr", "pal", "leucocytes"]
            presents_9 = [saisies[cle] is not None for cle in cles]
            valeurs_biomarqueurs = [
                np.log1p(saisies[cle]) if cle == "crp" and saisies[cle] is not None
                else (saisies[cle] if saisies[cle] is not None else 0.0)
                for cle in cles
            ]
            valeurs = np.array([age, est_femme] + valeurs_biomarqueurs, dtype=np.float32)
            age_biologique, risque = predire_avec_manquants(donnees_mlp, valeurs, presents_9)

        nb_present = sum(1 for bio in BIOMARQUEURS if saisies[bio["cle"]] is not None)
        colonne_1, colonne_2, colonne_3, colonne_4 = st.columns(4)
        colonne_1.metric("Âge biologique", f"{age_biologique:.1f} ans")
        colonne_2.metric("Écart avec l'âge réel", f"{age_biologique - age:+.1f} ans")
        colonne_3.metric("Risque de décès à 10 ans", f"{risque * 100:.1f} %")
        colonne_4.metric("Biomarqueurs utilisés", f"{nb_present}/9")
