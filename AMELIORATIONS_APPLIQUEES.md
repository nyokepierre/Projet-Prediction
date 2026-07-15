# Adaptations et améliorations appliquées

## Nouvelle cible

L’ancienne cible d’accessibilité structurelle ou potentielle a été remplacée par la cible finale de l’étude :

> recours effectif à une structure de santé formelle pour le cas de maladie le plus récent.

## Nouveau dispositif prédictif

- Modèle principal : XGBoost optimisé.
- Modèle complémentaire : Random Forest optimisé.
- 54 variables prédictives brutes.
- 319 colonnes après prétraitement.
- Seuil officiel de classement : 0,50.
- Suppression des variables postérieures au recours afin d’éviter la fuite de cible.

## Localisation automatique

Le choix de la commune met à jour la liste des quartiers. Le choix du quartier détermine automatiquement :

- la zone de résidence affichée à titre informatif ;
- la latitude médiane ;
- la longitude médiane.

Les coordonnées sont calculées à partir de la nouvelle base nettoyée.

## Formulaire amélioré

Le formulaire est organisé en sept onglets :

1. localisation ;
2. profil du ménage ;
3. situation économique ;
4. environnement ;
5. offre de soins et mobilité ;
6. besoin de soins ;
7. perception des services.

Les questions conditionnelles sont automatiquement masquées lorsque le ménage n’est pas assuré ou ne déclare pas une structure formelle proche.

## Résultats affichés

- probabilité de recours formel ;
- probabilité de non-recours ;
- ménage favorable ou ménage vulnérable ;
- niveau de vulnérabilité ;
- comparaison XGBoost / Random Forest ;
- explication individuelle SHAP ;
- rapport JSON ;
- historique CSV de la session.

## Protection des données

L’application ne demande aucun nom, numéro de téléphone ou identifiant personnel. Les coordonnées utilisées sont des valeurs médianes de quartier.
