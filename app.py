"""Application Streamlit : formulaire de biomarqueurs, estimation de l'age biologique."""
import numpy as np
import streamlit as st

from src.biomarqueurs import BIOMARQUEURS
from src.phenoage import calculer_phenoage
from src.logistique.logistique import predire_avec_manquants as predire_logistique_manquants
from src.mlp import charger_modele as charger_modele_mlp
from src.mlp.mlp import predire_avec_manquants as predire_mlp_manquants
from src.tabpfn.tabpfn import charger_modele_tabpfn, predire_pour_une_personne_tabpfn
from src.ft_transformer.ft_transformer import charger_modele_ft, predire_avec_manquants_ft

CLES = ["albumine", "creatinine", "glucose", "crp",
        "lymphocytes", "vgm", "idr", "pal", "leucocytes"]


def preparer_entree(sexe, saisies):
    """Prepare les entrees communes aux modeles ML/DL.

    Renvoie :
    - est_femme : 1.0 (femme), 0.0 (homme), 0.5 (non precise)
    - presents  : liste de 9 booleens (True si le biomarqueur est renseigne)
    - valeurs_9 : les 9 valeurs (CRP en log1p ; 0.0 si manquant)
    """
    est_femme = 1.0 if sexe == "Femme" else (0.0 if sexe == "Homme" else 0.5)
    presents = [saisies[cle] is not None for cle in CLES]
    valeurs_9 = []
    for cle in CLES:
        valeur = saisies[cle]
        if valeur is None:
            valeurs_9.append(0.0)
        elif cle == "crp":
            valeurs_9.append(np.log1p(valeur))
        else:
            valeurs_9.append(valeur)
    return est_femme, presents, valeurs_9


st.title("BioAge - Estimation de l'âge biologique")
st.caption("Outil de prévention, pas un diagnostic médical.")

modele_choisi = st.radio(
    "Modele d'estimation",
    options=[
        "PhenoAge (formule)",
        "Regression logistique (ML)",
        "MLP (deep learning)",
        "TabPFN",
        "FT-Transformer",
    ],
    horizontal=True,
    help="PhenoAge : formule de reference. "
         "Logistique : baseline ML. "
         "MLP : deep learning, bonne calibration. "
         "TabPFN : modele pre-entraine. "
         "FT-Transformer : modele de type transformer.",
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
            aides.append(f"Plage normale : {bio['plage_min']} - {bio['plage_max']} {bio['unite']}")
        if bio.get("plage_min_femme") is not None and bio.get("plage_max_femme") is not None:
            aides.append(f"Plage normale pour les femmes : "
                         f"{bio['plage_min_femme']} - {bio['plage_max_femme']} {bio['unite']}")
        if bio.get("plage_min_homme") is not None and bio.get("plage_max_homme") is not None:
            aides.append(f"\nPlage normale pour les hommes : "
                         f"{bio['plage_min_homme']} - {bio['plage_max_homme']} {bio['unite']}")
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
    elif manquants and modele_choisi.startswith("PhenoAge"):
        st.error("PhenoAge (formule) nécessite tous les biomarqueurs : " + ", ".join(manquants))
        st.info("Les modèles ML/DL (Logistique, MLP, TabPFN, FT-Transformer) acceptent les valeurs manquantes.")
    else:
        est_femme, presents, valeurs_9 = preparer_entree(sexe, saisies)
        valeurs_11 = np.array([age, est_femme] + valeurs_9, dtype=np.float32)

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
            age_biologique, risque = predire_logistique_manquants(valeurs_11, presents)
        elif modele_choisi.startswith("MLP"):
            age_biologique, risque = predire_mlp_manquants(charger_modele_mlp(), valeurs_11, presents)
        elif modele_choisi.startswith("TabPFN"):
            # TabPFN gere les NaN nativement : on remet les manquants a NaN
            valeurs_tabpfn = [v if present else np.nan for v, present in zip(valeurs_9, presents)]
            X_entree = np.array([age, est_femme] + valeurs_tabpfn, dtype=np.float32)
            age_biologique, risque = predire_pour_une_personne_tabpfn(charger_modele_tabpfn(), X_entree)
        else:  # FT-Transformer
            age_biologique, risque = predire_avec_manquants_ft(
                charger_modele_ft(), age, est_femme, valeurs_9, presents
            )

        nb_present = sum(presents)
        colonne_1, colonne_2, colonne_3, colonne_4 = st.columns(4)
        colonne_1.metric("Âge biologique", f"{age_biologique:.1f} ans")
        colonne_2.metric("Écart avec l'âge réel", f"{age_biologique - age:+.1f} ans")
        colonne_3.metric("Risque de décès à 10 ans", f"{risque * 100:.1f} %")
        colonne_4.metric("Biomarqueurs utilisés", f"{nb_present}/9")