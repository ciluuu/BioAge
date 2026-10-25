"""Fonctions pures pour creer des figures matplotlib.

Chaque fonction prend des donnees et renvoie une Figure. La sauvegarde est
l'affaire des scripts d'orchestration.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, confusion_matrix


def reliability_diagram(resultats_par_modele, n_bins=10):
    """Trace les courbes de calibration de plusieurs modeles sur un meme graphique.

    Parameters
    ----------
    resultats_par_modele : dict
        {nom_modele : {"probas": array, "y_vrai": array, "ece": float}}

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Calibration parfaite", alpha=0.5)
    for nom, donnees in resultats_par_modele.items():
        proba_moy, taux_reel = calibration_curve(
            donnees["y_vrai"], donnees["probas"], n_bins=n_bins
        )
        ax.plot(proba_moy, taux_reel, marker="o",
                label=f"{nom} (ECE={donnees['ece']:.3f})")
    ax.set_xlabel("Probabilite moyenne predite")
    ax.set_ylabel("Taux de deces reel")
    ax.set_title("Courbes de calibration (test NHANES IV)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def barres_comparaison_metriques(metriques_par_modele, metriques_a_tracer):
    """Barres groupees pour comparer plusieurs modeles sur plusieurs metriques.

    Parameters
    ----------
    metriques_par_modele : dict
        {nom_modele : {"auc": {"moyenne": x, "ecart_type": y}, ...}}
        ou {nom_modele : {"auc": float, ...}} (pas de barres d'erreur)
    metriques_a_tracer : list of str
        Noms des metriques, par ex. ["auc", "brier", "ece"]

    Returns
    -------
    matplotlib.figure.Figure
    """
    noms_modeles = list(metriques_par_modele.keys())
    nb_modeles = len(noms_modeles)
    x = np.arange(len(metriques_a_tracer))
    largeur = 0.8 / nb_modeles

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, nom_modele in enumerate(noms_modeles):
        valeurs = []
        erreurs = []
        for metrique in metriques_a_tracer:
            donnee = metriques_par_modele[nom_modele].get(metrique)
            if isinstance(donnee, dict):
                valeurs.append(donnee["moyenne"])
                erreurs.append(donnee["ecart_type"])
            else:
                valeurs.append(donnee)
                erreurs.append(0.0)
        ax.bar(x + i * largeur, valeurs, largeur, yerr=erreurs,
               label=nom_modele, capsize=4)

    ax.set_xticks(x + largeur * (nb_modeles - 1) / 2)
    ax.set_xticklabels(metriques_a_tracer)
    ax.set_ylabel("Valeur (plus bas = mieux sauf AUC)")
    ax.set_title("Comparaison des modeles sur le test NHANES IV")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def distribution_probas(probas, y_vrai, nom_modele=""):
    """Histogrammes des probabilites predites, separes par classe reelle."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(probas[y_vrai == 0], bins=50, alpha=0.6, label="Vivants a 10 ans", color="steelblue")
    ax.hist(probas[y_vrai == 1], bins=50, alpha=0.6, label="Decedes a 10 ans", color="darkred")
    ax.set_xlabel("Probabilite predite")
    ax.set_ylabel("Nombre de personnes")
    ax.set_title(f"Distribution des probabilites predites ({nom_modele})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def courbe_roc(resultats_par_modele):
    """Trace plusieurs courbes ROC sur un meme graphique.

    Parameters
    ----------
    resultats_par_modele : dict
        {nom : {"probas", "y_vrai", "auc"}}
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Hasard")
    for nom, donnees in resultats_par_modele.items():
        fpr, tpr, _ = roc_curve(donnees["y_vrai"], donnees["probas"])
        ax.plot(fpr, tpr, label=f"{nom} (AUC={donnees['auc']:.3f})")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbes ROC (test NHANES IV)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig