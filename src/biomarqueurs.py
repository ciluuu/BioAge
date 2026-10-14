"""Liste des 9 biomarqueurs utilisés par PhenoAge : nom, unité, exemple, plage valide."""

BIOMARQUEURS = [
    {"cle": "albumine",    "label": "Albumine",                      "unite": "g/L",     "exemple": 45.0, "min": 0.0, "plage_min" : 35.0, "plage_max" : 50.0},
    {"cle": "creatinine",  "label": "Créatinine",                    "unite": "µmol/L",  "exemple": 80.0, "min": 0.0, "plage_min" : None, "plage_max" : None, "plage_min_homme" : 60.0, "plage_max_homme" : 110.0, "plage_min_femme" : 45.0, "plage_max_femme" : 90.0},
    {"cle": "glucose",     "label": "Glucose",                       "unite": "mmol/L",  "exemple": 5.0,  "min": 0.0, "plage_min" : 3.9, "plage_max" : 5.5},
    {"cle": "crp",         "label": "Protéine C-réactive (CRP)",     "unite": "mg/L",    "exemple": 2.0,  "min": 0.0, "plage_min" : 0.0, "plage_max" : 3.0},
    {"cle": "lymphocytes", "label": "Lymphocytes",                   "unite": "%",       "exemple": 30.0, "min": 0.0, "plage_min" : 20.0, "plage_max" : 40.0},
    {"cle": "vgm",         "label": "Volume globulaire moyen (VGM)", "unite": "fL",      "exemple": 90.0, "min": 0.0, "plage_min" : 80.0, "plage_max" : 100.0},
    {"cle": "idr",         "label": "Indice de distribution (IDR)",  "unite": "%",       "exemple": 13.0, "min": 0.0, "plage_min" : 11.5, "plage_max" : 14.5},
    {"cle": "pal",         "label": "Phosphatases alcalines (PAL)",  "unite": "U/L",     "exemple": 70.0, "min": 0.0, "plage_min" : 40.0, "plage_max" : 130.0},
    {"cle": "leucocytes",  "label": "Leucocytes",                    "unite": "1000/µL", "exemple": 6.0,  "min": 0.0, "plage_min" : 4.0, "plage_max" : 10.0},
]