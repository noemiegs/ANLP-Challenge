"""
Ce script permet de réaliser l'entraînement d'un modèle de classification de texte (reconnaissance d'une langue)
"""

### data_preprocessing.py
import pandas as pd
import random
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
### model_training.py
import torch
import mlflow
import mlflow.transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import json
from sklearn.model_selection import KFold, StratifiedKFold
import nlpaug.augmenter.word as naw

# Paramètres globaux
EARLY_STOPPING = True
CROSS_VALIDATION = False
DATA_AUGMENTATION = True
REMOVE_OCCURENCES = True
REM_URL = False

# data
DATA_PATH = "data/train_submission.csv"

# inference 
TEST_PATH = "data/test_without_labels.csv"
LABEL_MAPPING_PATH = "data/mapping/test_intfloat_mappings.json"
OUTPUT_PATH = "submission.csv"
CKPT_PATH = '/home/ecstatic_easley/ANLP-Challenge/models/test_intfloat/checkpoint-9401' # si chargement d'un modèle pré-entraîné, inférence uniquement dans ce cas

# paramètres d'entraînement
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
MODEL_PATH_NAME = "models/test_intfloat"
TRAIN_EPOCHS = 1
LEARNING_RATE = 3e-5
BATCH_SIZE = 128
STRATIFY = True
RANDOM_STATE = True # True pour fixer le split des données
CV_SPLITS = 3

# métriques
LOG_FILE_PATH = "logs/model_results.csv"
LOG_METRICS = True



def remove_urls(data):
    if REM_URL:
        url_pattern = r'https://\S+'
        data['Text'] = data['Text'].apply(lambda x: re.sub(url_pattern, '', str(x)))


def augment_data(data):
    if DATA_AUGMENTATION:
        label_counts = data.Label.value_counts()
        data_aug = pd.DataFrame(columns=['Text', 'Label'])
        for label, count in label_counts[label_counts < 20].items():
            label_data = data[data['Label'] == label]
            idx = 0
            while len(label_data) + len(data_aug[data_aug['Label'] == label]) < 20:
                example = label_data.iloc[idx]['Text']
                words = example.split()
                random.shuffle(words)
                new_example = ' '.join(words)
                data_aug = pd.concat([data_aug, pd.DataFrame({'Text': [new_example], 'Label': [label]})], ignore_index=True)
                idx = (idx + 1) % len(label_data)
        return pd.concat([data, data_aug], ignore_index=True)
    return data



# Ajout d'un augmentateur pour les fautes de frappe, suppression de mots, et échange de mots
def get_augmented_examples(example, augmenter_typo, augmenter_delete, augmenter_swap):
    """
    Applique différentes augmentations au texte de manière aléatoire.
    """
    # Décider aléatoirement d'appliquer certaines transformations
    if random.random() < 0.3:  # 30% de chance pour une faute de frappe
        example = augmenter_typo.augment(example)
    
    if random.random() < 0.3:  # 30% de chance de supprimer un mot
        example = augmenter_delete.augment(example)
    
    if random.random() < 0.4:  # 30% de chance de permuter deux mots
        example = augmenter_swap.augment(example)
    
    return example

# def augment_data(data):
#     """
#     Applique des augmentations de données pour augmenter la diversité du texte.
#     Cette fonction se concentre sur les labels ayant moins de 10 exemples.
#     """
#     if DATA_AUGMENTATION:
#         label_counts = data.Label.value_counts()
#         data_aug = pd.DataFrame(columns=['Text', 'Label'])
        
#         # Initialisation des augmentateurs
#         augmenter_typo = naw.SpellingAug(aug_p=0.1)  # Fautes de frappe dans 10% du texte
#         augmenter_delete = naw.RandomWordAug(action="delete",aug_p=0.1)  # Suppression aléatoire de mots
#         augmenter_swap = naw.RandomWordAug(action="swap", aug_p=0.1)  # Permutation de mots
        
#         # Appliquer l'augmentation uniquement aux labels ayant moins de 10 exemples
#         for label, count in label_counts[label_counts < 50].items():
#             label_data = data[data['Label'] == label]
#             idx = 0
#             while len(label_data) + len(data_aug[data_aug['Label'] == label]) < 50:
#                 example = label_data.iloc[idx]['Text']
#                 new_example = get_augmented_examples(example, augmenter_typo, augmenter_delete, augmenter_swap)
#                 data_aug = pd.concat([data_aug, pd.DataFrame({'Text': [new_example], 'Label': [label]})], ignore_index=True)
#                 idx = (idx + 1) % len(label_data)
        
#         # Retourner les données augmentées
#         return pd.concat([data, data_aug], ignore_index=True)
    
#     return data


def split_data(data, use_stratify=STRATIFY):
    data['LabelID'] = pd.factorize(data['Label'])[0]
    stratify = data['LabelID'] if use_stratify else None 
    random_state = 42 if RANDOM_STATE else None
    train_texts, test_texts, train_labels, test_labels = train_test_split(data['Text'], data['LabelID'], test_size=0.1, stratify=stratify, random_state=random_state)
    val_texts, test_texts, val_labels, test_labels = train_test_split(test_texts, test_labels, test_size=0.5, stratify=test_labels if stratify is not None else None, random_state=random_state)
    return DatasetDict({
        'train': Dataset.from_dict({'Text': train_texts.tolist(), 'LabelID': train_labels.tolist()}),
        'validation': Dataset.from_dict({'Text': val_texts.tolist(), 'LabelID': val_labels.tolist()}),    # comment for final training
        'test': Dataset.from_dict({'Text': test_texts.tolist(), 'LabelID': test_labels.tolist()})         # comment for final training
    })



def cross_validate(data, n_splits=3, stratify=STRATIFY):
    data['LabelID'] = pd.factorize(data['Label'])[0]
    if stratify: 
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    fold = 1

    for train_idx, val_idx in kf.split(data['Text'], data['LabelID']):
        print(f"🚨 Fold {fold}/{n_splits}")

        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]

        dataset = DatasetDict({
            'train': Dataset.from_dict({'Text': train_data['Text'].tolist(), 'LabelID': train_data['LabelID'].tolist()}),
            'validation': Dataset.from_dict({'Text': val_data['Text'].tolist(), 'LabelID': val_data['LabelID'].tolist()})
        })

        trainer = NLPTrainer(dataset)
        trainer.train()
        result = trainer.evaluate()
        results.append(result)
        fold += 1

    # Moyenne des métriques sur tous les folds
    avg_results = {k: sum(d[k] for d in results) / len(results) for k in results[0]}
    print(f"🏆 Cross-validation moyenne : {avg_results}")
    return avg_results





mlflow.set_experiment(MODEL_PATH_NAME)

def compute_metrics(pred):
    preds = pred.predictions.argmax(axis=1)
    labels = pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return { 'accuracy': accuracy_score(labels, preds), 'f1': f1, 'precision': precision, 'recall': recall }


def log_results(params, metrics):
    if LOG_METRICS:
        result_line = {**params, **metrics}
        log_exists = os.path.exists(LOG_FILE_PATH)
        result_df = pd.DataFrame([result_line])
        result_df.to_csv(LOG_FILE_PATH, mode='a', header=not log_exists, index=False)


class NLPTrainer:
    def __init__(self, dataset, ckpt_path=None):
        """
        :param dataset: Le dataset d'entraînement.
        :param ckpt_path: Chemin vers un modèle pré-entraîné. Si fourni, charge ce modèle.
        """
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model_path = f"./{MODEL_PATH_NAME}" if MODEL_PATH_NAME else None
        
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"🔄 Chargement du modèle pré-entraîné depuis: {ckpt_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
        else:
            print("🏋️ Nouveau modèle chargé pour l'entraînement")
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(set(dataset['train']['LabelID'])))

    def tokenize(self):
        """Applique la tokenisation sur le dataset"""
        def tokenize_fn(batch):
            tokens = self.tokenizer(batch['Text'], truncation=True, padding=True, max_length=100)
            tokens["labels"] = batch["LabelID"]
            return tokens

        return self.dataset.map(tokenize_fn, batched=True)

    def train(self):
        """Entraîne le modèle"""
        tokenized_data = self.tokenize()

        # Définir les arguments d'entraînement
        args = TrainingArguments(
            output_dir=self.model_path,
            overwrite_output_dir=True,
            evaluation_strategy="epoch", # comment for final training
            save_strategy="epoch",
            save_only_model=True,
            save_total_limit=2,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            num_train_epochs=TRAIN_EPOCHS,
            load_best_model_at_end=True,  # comment for final training
            metric_for_best_model="accuracy",  # Utilise l'accuracy pour déterminer le meilleur modèle
            greater_is_better=True,
            fp16=torch.cuda.is_available()
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_data['train'],
            eval_dataset=tokenized_data['validation'], # comment for final training
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if EARLY_STOPPING else []
        )

        mlflow.transformers.autolog()
        trainer.train()

        if self.model_path:
            # Sauvegarder le meilleur modèle
            self.model.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.model_path)
            print(f"✅ Modèle sauvegardé dans {self.model_path}")

    def evaluate(self):
        """Évalue le modèle"""
        tokenized_data = self.tokenize()
        trainer = Trainer(model=self.model)
        test_results = trainer.evaluate(tokenized_data["test"])
        return test_results

    def inference_submission(self, test_path=TEST_PATH, mapping_path=LABEL_MAPPING_PATH, output_file=OUTPUT_PATH):
        """Fait l'inférence et génère la soumission"""
        # Charger le fichier de test
        df = pd.read_csv(test_path).drop(columns='Usage', errors='ignore')

        # Tokenisation des textes
        inputs = self.tokenizer(df["Text"].tolist(), truncation=True, padding=True, max_length=100, return_tensors="pt")

        # Déplacer le modèle sur le GPU si dispo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        # Créer DataLoader
        dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
        dataloader = DataLoader(dataset, batch_size=128)

        predictions = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="🔍 Running Inference", unit="batch"):
                batch = [tensor.to(device) for tensor in batch]
                outputs = self.model(input_ids=batch[0], attention_mask=batch[1])
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(preds)

        # Charger les mappings
        with open(mapping_path, "r") as f:
            mappings = json.load(f)
        id2label = {int(k): v for k, v in mappings["id2label"].items()}

        # Mapper les prédictions
        df["Label"] = predictions
        df["Label"] = df["Label"].map(id2label)
        df["ID"] = range(1, len(df) + 1)

        # Sauvegarder le fichier de soumission
        df = df[["ID", "Label"]]
        if os.path.exists(output_file):
            print(f"❌ Le fichier {output_file} existe déjà.")
        else:
            df.to_csv(output_file, index=False)
            print(f"✅ Fichier de soumission sauvegardé: {output_file}")



if __name__ == '__main__':
    # Chargement des données et préparation
    data = pd.read_csv(DATA_PATH, keep_default_na=False).dropna()
    data = augment_data(data)
    if CROSS_VALIDATION:
        test_results = cross_validate(data)  # Récupère les résultats de la CV
        print("Résultats finaux CV:", test_results)
    else:
        dataset = split_data(data)
        trainer = NLPTrainer(dataset, ckpt_path=CKPT_PATH)
        if CKPT_PATH is None:
            trainer.train()
        # test_results = trainer.evaluate()

    if CKPT_PATH is None:  # Log uniquement si on a entraîné un modèle
        log_results({
            'MODEL_NAME': MODEL_NAME,
            'EARLY_STOPPING': EARLY_STOPPING,
            'CROSS_VALIDATION': CROSS_VALIDATION,
            'DATA_AUGMENTATION': DATA_AUGMENTATION,
            'REMOVE_OCCURENCES': REMOVE_OCCURENCES,
            'LEARNING_RATE': LEARNING_RATE,
            'TRAIN_EPOCHS': TRAIN_EPOCHS,
            'BATCH_SIZE': BATCH_SIZE
        }, test_results)
    trainer.inference_submission()
