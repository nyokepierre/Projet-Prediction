# Guide d’installation de la nouvelle interface

Cette mise à jour concerne uniquement la présentation de l’application. Elle ne
modifie ni le modèle entraîné, ni le préprocesseur, ni `app_config.json`.

## Fichiers à remplacer

1. Remplacer le fichier existant `app.py` par le nouveau `app.py`.
2. Remplacer `.streamlit/config.toml` par la nouvelle version.

## Fichiers à ajouter

1. Ajouter `ui_components.py` à la racine du projet, au même niveau que
   `app.py`.
2. Créer un dossier `assets` à la racine du projet.
3. Ajouter `assets/styles.css` dans ce dossier.

La structure finale doit être la suivante :

```text
votre-projet/
├── app.py
├── ui_components.py
├── app_config.json
├── requirements.txt
├── modeles/
│   └── modele_acces_soins_bundle.joblib
├── assets/
│   └── styles.css
└── .streamlit/
    └── config.toml
```

## Lancement

```bash
pip install -r requirements.txt
python smoke_test.py
streamlit run app.py
```

## Corrections graphiques appliquées

- suppression du sélecteur CSS global qui imposait une couleur blanche à tous
  les textes de la barre latérale ;
- contraste renforcé entre textes, champs et arrière-plans ;
- barre latérale claire et lisible ;
- menus déroulants avec texte sombre sur fond blanc ;
- meilleure lisibilité des champs désactivés ;
- boutons principaux et secondaires clairement différenciés ;
- en-tête professionnel et responsive ;
- cartes de résultats avec états favorable et vulnérable très contrastés ;
- onglets, métriques, alertes, graphiques et tableaux harmonisés ;
- comportement responsive amélioré sur tablette et téléphone.

## Mise à jour sur Streamlit Cloud

Après avoir remplacé les fichiers dans le dépôt GitHub :

1. valider les modifications avec un nouveau commit ;
2. pousser le commit sur GitHub ;
3. redémarrer l’application depuis Streamlit Cloud si l’ancien thème reste en
   cache.
