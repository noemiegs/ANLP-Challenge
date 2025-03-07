import pandas as pd
import os
from model import NLPTrainer, split_data, augment_data, remove_urls, cross_validate
from transformers import set_seed

# Paramètres globaux
DATA_PATH = "data/train_submission.csv"
MODELS = [
    # "intfloat/multilingual-e5-large-instruct",
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
    # "distilbert-base-multilingual-cased"
]

EARLY_STOPPING = True
BATCH_SIZE = 64,
# "STRATIFY": True, on ne peut utiliser la stratification sans DATA_AUGMENTATION
TRAIN_EPOCHS = 1,
LEARNING_RATE = 3e-5
RANDOM_STATE = True
CKPT_PATH = '/home/ecstatic_easley/ANLP-Challenge/models/test_intfloat/checkpoint-9401'

CONFIG_TESTS = [
    {"name": "Baseline", "CROSS_VALIDATION": False, "DATA_AUGMENTATION": False, "REMOVE_OCCURENCES": False, "REM_URL": False, "STRATIFY": False},
    {"name": "Data Augmentation", "CROSS_VALIDATION": False,"DATA_AUGMENTATION": True, "REMOVE_OCCURENCES": False, "REM_URL": False, "STRATIFY": False},
    {"name": "Data Augmentation + Stratify", "CROSS_VALIDATION": False,"DATA_AUGMENTATION": True, "REMOVE_OCCURENCES": False, "REM_URL": False, "STRATIFY": True},
    {"name": "Remove Occurrences", "CROSS_VALIDATION": False,"DATA_AUGMENTATION": False, "REMOVE_OCCURENCES": True, "REM_URL": False, "STRATIFY":
    False},
    {"name": "Remove URLs", "CROSS_VALIDATION": False, "DATA_AUGMENTATION": False, "REMOVE_OCCURENCES": False, "REM_URL": True, "STRATIFY": False},
    {"name": "Cross Validation", "CROSS_VALIDATION": True, "DATA_AUGMENTATION": False, "REMOVE_OCCURENCES": False, "REM_URL": False, "STRATIFY": False, "CV_SPLITS": 3}
]

RESULTS_FILE_MODEL = "logs/model_comparison_results.csv"
RESULTS_FILE_CONFIG = "logs/config_tests_results.csv"

def run_model_comparison():
    """ Test rapide pour comparer plusieurs modèles """
    print("Comparaison de modèles...")
    data = pd.read_csv(DATA_PATH).dropna()
    dataset = split_data(data, use_stratify=False)

    results = []

    for model_name in MODELS:
        print(f"🚀 Test du modèle: {model_name}")
        NLPTrainer.MODEL_NAME = model_name
        NLPTrainer.MODEL_PATH_NAME = f"models/{model_name.split('/')[-1]}"
        trainer = NLPTrainer(dataset, CKPT_PATH)
        if CKPT_PATH is None:
                trainer.train()
        metrics = trainer.evaluate()
        print(f"Résultats: {metrics}")
        print(metrics.keys())
        # results.append({
        #     "Model": model_name,
        #     "Accuracy": metrics["accuracy"],
        #     "F1": metrics["f1"]
        # })

    pd.DataFrame(results).to_csv(RESULTS_FILE_MODEL, index=False)
    print(f"Résultats sauvegardés dans {RESULTS_FILE_MODEL}")


# def run_config_tests():
    """ Test des différentes configurations sur le modèle prometteur """
    print("Test des configurations...")
    data = pd.read_csv(DATA_PATH).dropna()

    for config in CONFIG_TESTS:
        print(f"Test config: {config['name']}")
        data_aug = augment_data(data) if config["DATA_AUGMENTATION"] else data
        if config["REMOVE_OCCURENCES"]:
            data_aug = data_aug.drop_duplicates(subset=['Text'])
        if config["REM_URL"]:
            remove_urls(data_aug)
        
        # Fixer la seed
        set_seed(42)

        model_name = "intfloat/multilingual-e5-large-instruct"
        model_path = f"models/{model_name.split('/')[-1]}"
        
        # Utilisation de cross-validation si la configuration le demande
        if merged_config["CROSS_VALIDATION"]:
            # Si CROSS_VALIDATION est True, utiliser la méthode de CV
            metrics = cross_validate(data_aug, n_splits=merged_config["CV_SPLITS"], stratify=merged_config["STRATIFY"])
            results.append({
                "Config": merged_config["name"],
                "Accuracy": metrics["accuracy"],
                "F1": metrics["f1"]
            })
        else:
            dataset = split_data(data_aug, use_stratify=config["STRATIFY"])
            trainer = NLPTrainer(dataset, model_name=model_name, model_path=model_path)
            trainer.train()
            metrics = trainer.evaluate()
            print(f"📊 Résultats: {metrics}")
            print(metrics.keys())
            results.append({
                "Config": merged_config["name"],
                "Accuracy": metrics["accuracy"],
                "F1": metrics["f1"]
            })

    pd.DataFrame(results).to_csv(RESULTS_FILE_CONFIG, mode='a', header=False, index=False)
    print(f"✅ Résultats des configs sauvegardés dans {RESULTS_FILE_CONFIG}")


if __name__ == "__main__":
    run_model_comparison()
    # run_config_tests()
