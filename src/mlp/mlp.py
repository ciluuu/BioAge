"""MLP PyTorch et fonctions pures pour la prediction de mortalite a 10 ans."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from src.conversion import risque_vers_age_biologique

NB_FEATURES = 11                                   # age + sexe + 9 biomarqueurs
CHEMIN_MODELE = Path("models/mlp.pt")


class MLP(nn.Module):
    """Reseau 22 -> 64 -> 32 -> 1 logit. Classe imposee par PyTorch."""

    def __init__(self, dropout=0.2):
        super().__init__()
        self.reseau = nn.Sequential(
            nn.Linear(NB_FEATURES * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.reseau(x)


def construire_modele(dropout=0.2, graine=42):
    """Fabrique un MLP neuf avec graine fixee pour reproductibilite."""
    torch.manual_seed(graine)
    return MLP(dropout=dropout)


def standardiser(X_reference, X_a_transformer):
    """Centre-reduit X_a_transformer avec les stats de X_reference."""
    moyennes = X_reference.mean(axis=0)
    ecarts_types = X_reference.std(axis=0) + 1e-8
    return (X_a_transformer - moyennes) / ecarts_types, moyennes, ecarts_types


def masquer_aleatoire(X, taux_max=3, graine=None):
    """Masque aleatoirement 0 a taux_max biomarqueurs par ligne.

    L'age (col 0) et le sexe (col 1) ne sont jamais masques.
    Renvoie une matrice (X_masque | masque) de forme (n, 2*NB_FEATURES).
    """
    rng = np.random.default_rng(graine)
    X_masque = X.copy()
    masque = np.ones_like(X)
    nb_a_masquer = rng.integers(0, taux_max + 1, size=len(X))
    for i, k in enumerate(nb_a_masquer):
        if k > 0:
            idx = rng.choice(range(2, NB_FEATURES), size=k, replace=False)
            X_masque[i, idx] = 0.0
            masque[i, idx] = 0.0
    return np.concatenate([X_masque, masque], axis=1).astype(np.float32)


def ajouter_masque_complet(X):
    """Concatene un masque tout a 1 (donnees completes, pas de masquage)."""
    masque = np.ones_like(X)
    return np.concatenate([X, masque], axis=1).astype(np.float32)


def predire_probas(modele, X_prepare):
    """Applique sigmoid aux logits, renvoie toujours un array 1D (meme pour batch=1)."""
    modele.eval()
    with torch.no_grad():
        logits = modele(torch.from_numpy(X_prepare)).squeeze(-1).numpy()
    probas = 1.0 / (1.0 + np.exp(-logits))
    return np.atleast_1d(probas)


def entrainer_une_epoch(modele, X_entree, y, batch_size, optimiseur, loss_fn):
    """Fait passer un epoch complet et renvoie la perte moyenne."""
    dataset = TensorDataset(
        torch.from_numpy(X_entree),
        torch.from_numpy(y).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    modele.train()
    pertes = [
        _un_pas(modele, X_batch, y_batch, optimiseur, loss_fn)
        for X_batch, y_batch in loader
    ]
    return float(np.mean(pertes))


def _un_pas(modele, X_batch, y_batch, optimiseur, loss_fn):
    """Un pas de descente de gradient. Encapsule la mutation."""
    optimiseur.zero_grad()
    logits = modele(X_batch)
    perte = loss_fn(logits, y_batch)
    perte.backward()
    optimiseur.step()
    return perte.item()


def entrainer_avec_early_stopping(X_tr_std, y_tr, X_val_std, y_val,
                                   nb_epochs=100, patience=10,
                                   batch_size=128, lr=1e-3,
                                   weight_decay=1e-5, dropout=0.2):
    """Boucle d'entrainement avec early stopping sur l'AUC validation.

    Renvoie un modele neuf avec les meilleurs poids restaures.
    """
    modele = construire_modele(dropout=dropout)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    X_val_prepare = ajouter_masque_complet(X_val_std)

    meilleure_auc = 0.0
    meilleur_etat = None
    patience_restante = patience

    for epoch in range(nb_epochs):
        X_tr_prepare = masquer_aleatoire(X_tr_std, graine=epoch)
        entrainer_une_epoch(modele, X_tr_prepare, y_tr, batch_size, optimiseur, loss_fn)
        probas_val = predire_probas(modele, X_val_prepare)
        auc = roc_auc_score(y_val, probas_val)

        if auc > meilleure_auc:
            meilleure_auc = auc
            meilleur_etat = {cle: v.clone() for cle, v in modele.state_dict().items()}
            patience_restante = patience
        else:
            patience_restante -= 1
            if patience_restante == 0:
                break

    modele.load_state_dict(meilleur_etat)
    return modele


def sauvegarder_modele(modele, moyennes, ecarts_types, chemin=CHEMIN_MODELE):
    """Serialise le modele et les stats de standardisation."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "poids": modele.state_dict(),
        "moyennes": moyennes,
        "ecarts_types": ecarts_types,
    }, chemin)


def charger_modele(chemin=CHEMIN_MODELE):
    """Fonction pure : prend un chemin, renvoie un dict. Pas de cache global."""
    contenu = torch.load(chemin, map_location="cpu", weights_only=False)
    modele = construire_modele()
    modele.load_state_dict(contenu["poids"])
    modele.eval()
    return {
        "modele": modele,
        "moyennes": contenu["moyennes"],
        "ecarts_types": contenu["ecarts_types"],
    }


def predire_pour_une_personne(donnees_modele, valeurs_11):
    """Predit l'age biologique et le risque pour une personne."""
    valeurs_std = (valeurs_11 - donnees_modele["moyennes"]) / donnees_modele["ecarts_types"]
    X = ajouter_masque_complet(valeurs_std.reshape(1, -1))
    risque = float(predire_probas(donnees_modele["modele"], X)[0])
    return risque_vers_age_biologique(risque), risque

def predire_avec_manquants(donnees_modele, valeurs_11, presents_9):
    """Predit en gerant les biomarqueurs manquants via le masque natif.

    Parameters
    ----------
    donnees_modele : dict renvoye par charger_modele
    valeurs_11 : ndarray de shape (11,)
        age, est_femme, puis 9 biomarqueurs (0.0 pour les manquants).
    presents_9 : list[bool] de longueur 9
        True si le biomarqueur est renseigne.

    Returns
    -------
    age_biologique, risque_10_ans : float
    """
    valeurs = valeurs_11.copy()
    masque = np.ones(NB_FEATURES, dtype=np.float32)
    masque[2:] = np.array(presents_9, dtype=np.float32)      # age et sexe toujours la

    # Standardisation puis masquage a zero des positions manquantes
    valeurs_std = (valeurs - donnees_modele["moyennes"]) / donnees_modele["ecarts_types"]
    valeurs_std = valeurs_std * masque                       # 0 ou l'on n'a pas la valeur

    X = np.concatenate([valeurs_std, masque]).reshape(1, -1).astype(np.float32)
    risque = float(predire_probas(donnees_modele["modele"], X)[0])
    return risque_vers_age_biologique(risque), risque