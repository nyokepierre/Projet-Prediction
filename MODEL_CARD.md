# Fiche du modèle

## Cible

Recours effectif à une structure de santé formelle pour le cas de maladie le
plus récent :

- 1 : recours formel ;
- 0 : non-recours formel.

## Données

- 1 000 ménages ;
- 599 recours formels ;
- 401 non-recours ;
- 54 prédicteurs bruts ;
- 319 colonnes après prétraitement ;
- séparation apprentissage/test : 80/20 stratifiée.

## Modèle principal

XGBoost optimisé.

- AUC-ROC test : 0.793
- Exactitude test : 71.0%
- Exactitude équilibrée : 68.5%
- Rappel du non-recours : 56.2%
- PR-AUC du non-recours : 0.702
- Score de Brier : 0.182

## Modèle complémentaire

Random Forest optimisé, utilisé comme contrôle de robustesse.

- AUC-ROC test : 0.786
- Exactitude test : 72.0%
- Rappel du non-recours : 67.5%

## Limites

- Les performances diminuent lorsque le modèle est appliqué à des communes
  entièrement absentes de l’apprentissage.
- Les contributions SHAP décrivent la logique prédictive et non une causalité.
- Les informations géographiques ont une influence importante sur le modèle.
- Toute utilisation hors de Kinshasa exige une validation externe.
