"""Liste des 9 biomarqueurs utilisés par PhenoAge : nom, unité, exemple, plage valide."""

BIOMARQUEURS = [
    {"cle": "albumine",    "label": "Albumine",                      "unite": "g/L",     "exemple": 45.0, "min": 20.0, "max": 60.0},
    {"cle": "creatinine",  "label": "Créatinine",                    "unite": "µmol/L",  "exemple": 80.0, "min": 20.0, "max": 500.0},
    {"cle": "glucose",     "label": "Glucose",                       "unite": "mmol/L",  "exemple": 5.0,  "min": 2.0,  "max": 30.0},
    {"cle": "crp",         "label": "Protéine C-réactive (CRP)",     "unite": "mg/L",    "exemple": 2.0,  "min": 0.1,  "max": 200.0},
    {"cle": "lymphocytes", "label": "Lymphocytes",                   "unite": "%",       "exemple": 30.0, "min": 1.0,  "max": 80.0},
    {"cle": "vgm",         "label": "Volume globulaire moyen (VGM)", "unite": "fL",      "exemple": 90.0, "min": 60.0, "max": 120.0},
    {"cle": "idr",         "label": "Indice de distribution (IDR)",  "unite": "%",       "exemple": 13.0, "min": 10.0, "max": 25.0},
    {"cle": "pal",         "label": "Phosphatases alcalines (PAL)",  "unite": "U/L",     "exemple": 70.0, "min": 20.0, "max": 500.0},
    {"cle": "leucocytes",  "label": "Leucocytes",                    "unite": "1000/µL", "exemple": 6.0,  "min": 1.0,  "max": 30.0},
]