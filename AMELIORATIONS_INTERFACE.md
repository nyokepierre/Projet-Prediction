# Améliorations de l’interface graphique

## Problème principal corrigé

L’ancienne feuille de style contenait la règle suivante :

```css
[data-testid="stSidebar"] * {
    color: #ffffff;
}
```

Cette règle appliquait le blanc à tous les éléments de la barre latérale, y
compris certains textes placés sur des fonds clairs. Elle provoquait donc des
zones illisibles. La nouvelle version cible séparément les titres, les textes,
les champs, les boutons et les menus.

## Système visuel retenu

- fond général gris très clair ;
- surfaces blanches ;
- texte principal bleu nuit ;
- couleur d’action vert sarcelle ;
- vert pour les résultats favorables ;
- rouge sombre pour les profils vulnérables ;
- bordures discrètes et ombres légères ;
- hiérarchie typographique renforcée.

## Composants ajoutés

Le fichier `ui_components.py` centralise :

- le chargement du thème ;
- l’en-tête principal ;
- l’identité visuelle de la barre latérale ;
- les badges du modèle ;
- les titres d’étapes ;
- la bannière du résultat ;
- le pied de page.

Cette organisation rend le code principal plus propre et facilite les futures
modifications de l’interface.
