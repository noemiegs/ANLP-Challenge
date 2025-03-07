# ANLP-Challenge

Ce projet vise à développer un classifieur de texte dans le cadre du cours *Advanced Natural Language Processing*. L'objectif est de classifier des textes selon leur langue.

## Installation

Avant d'exécuter le code, assurez-vous d'installer les dépendances requises :

```bash
pip install -r requirements.txt
```

## Structure du projet
Le projet est organisé en plusieurs fichiers permettant l'exploration des données, l'entraînement des modèles et la soumission des prédictions.

- **`0-eda.ipynb`** : Notebook contenant l'analyse exploratoire des données (EDA), permettant de comprendre la distribution des classes, la longueur des textes, et les valeurs manquantes.
- **`1-model_intfloat.ipynb`** : Notebook utilisé pour entraîner et tester notre modèle final basé sur `intfloat/multilingual-e5-large-instruct`.
- **`2-model_distilbert.ipynb`** : Expérimentations menées avec un autre modèle, `distilbert-base-multilingual-cased`, pour comparer les performances.
- **`3-run_test_to_submit.ipynb`** : Génère le fichier de soumission pour Kaggle en effectuant une inférence sur le jeu de test.
- **`model.py`** : Interface permettant de lancer un entraînement et une inférence avec un modèle et des hyperparamètres personnalisés.
- **`model_comparison.py`** : Permet d'effectuer une *grid search*, tester différents modèles et configurations pour identifier la meilleure architecture.

## Résultats et Performances

Les résultats obtenus sont évalués avec des métriques classiques comme :

- **Accuracy**
- **F1-score**
- **Precision**
- **Recall**

Les résultats finaux des modèles sont sauvegardés dans :

- `logs/model_comparison_results.csv` : Résultats de la comparaison des modèles.
- `logs/config_tests_results.csv` : Résultats des différentes configurations testées.

## Compétition Kaggle

Les résultats finaux sont soumis sur la plateforme Kaggle dans le cadre du challenge *NLP CS 2025*.

La compétition est disponible ici : [Kaggle - NLP CS 2025](https://www.kaggle.com/competitions/nlp-cs-2025)




