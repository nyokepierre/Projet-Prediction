# Application de prédiction du recours formel aux soins

Cette version adapte l’ancienne application à la nouvelle configuration de
l’étude et à la base analytique nettoyée de 1 000 ménages.

## Principales modifications

- La cible est désormais le **recours effectif à une structure de santé formelle**.
- Le modèle principal est le **XGBoost optimisé**.
- Le Random Forest est utilisé comme contrôle complémentaire de robustesse.
- L’application utilise exactement les **54 prédicteurs** du protocole final.
- Les variables postérieures à la décision de recours ont été supprimées afin
  d’éviter la fuite de cible.
- La commune détermine automatiquement :
  - la liste des quartiers disponibles ;
  - la zone de résidence affichée à titre informatif ;
  - la latitude et la longitude médianes du quartier sélectionné.
- Les questions conditionnelles sont masquées lorsque leur contexte ne
  s’applique pas :
  - couverture santé ;
  - caractéristiques de la structure la plus proche.
- Le résultat affiche :
  - la probabilité de recours ;
  - la probabilité de non-recours ;
  - le profil « ménage favorable » ou « ménage vulnérable » ;
  - la comparaison XGBoost / Random Forest ;
  - une explication locale SHAP ;
  - un rapport JSON téléchargeable.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Vérifier les fichiers du modèle

```bash
python smoke_test.py
```

## Lancer l’application

```bash
streamlit run app.py
```

## Déploiement Streamlit Community Cloud

1. Déposer tous les fichiers de ce dossier dans un dépôt GitHub.
2. Vérifier que le fichier
   `modeles/modele_acces_soins_bundle.joblib` est bien présent.
3. Sur Streamlit Community Cloud, sélectionner `app.py`.
4. Utiliser une version de Python compatible avec les dépendances indiquées
   dans `requirements.txt`.

## Interprétation

Le seuil officiel est fixé à 0,50 :

- probabilité de recours >= 0,50 : **ménage favorable** ;
- probabilité de recours < 0,50 : **ménage vulnérable**.

Le classement est une estimation statistique. Il ne constitue ni un diagnostic
médical, ni une décision administrative ou sociale automatique.

## Confidentialité

L’application ne nécessite pas de nom, de numéro de téléphone ou d’autre
identifiant personnel. Les coordonnées utilisées sont des médianes de quartier
et non l’adresse exacte du ménage.
