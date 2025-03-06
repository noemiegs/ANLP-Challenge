### config.py
# Paramètres globaux

EARLY_STOPPING = True
CROSS_VALIDATION = False
DATA_AUGMENTATION = True
REMOVE_OCCURENCES = True
REM_URL = False

DATA_PATH = "data/train_submission.csv"
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
MODEL_PATH_NAME = "test_intfloat"
TRAIN_EPOCHS = 5
LEARNING_RATE = 3e-5
BATCH_SIZE = 32
STRATIFY = True
LOG_FILE_PATH = "model_results.csv"
LOG_METRICS = True
ALLOW_RETRAIN = True  # Autoriser ou non le retrain avec les mêmes paramètres

### data_preprocessing.py
import pandas as pd
import random
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from config import *

def remove_urls(data):
    if REM_URL:
        url_pattern = r'https://\S+'
        data['Text'] = data['Text'].apply(lambda x: re.sub(url_pattern, '', str(x)))


def augment_data(data):
    if DATA_AUGMENTATION:
        label_counts = data.Label.value_counts()
        data_aug = pd.DataFrame(columns=['Text', 'Label'])
        for label, count in label_counts[label_counts < 10].items():
            label_data = data[data['Label'] == label]
            idx = 0
            while len(label_data) + len(data_aug[data_aug['Label'] == label]) < 10:
                example = label_data.iloc[idx]['Text']
                words = example.split()
                random.shuffle(words)
                new_example = ' '.join(words)
                data_aug = pd.concat([data_aug, pd.DataFrame({'Text': [new_example], 'Label': [label]})], ignore_index=True)
                idx = (idx + 1) % len(label_data)
        return pd.concat([data, data_aug], ignore_index=True)
    return data


def split_data(data):
    data['LabelID'] = pd.factorize(data['Label'])[0]
    stratify = data['LabelID'] if STRATIFY else None
    train_texts, test_texts, train_labels, test_labels = train_test_split(data['Text'], data['LabelID'], test_size=0.2, stratify=stratify)
    val_texts, test_texts, val_labels, test_labels = train_test_split(test_texts, test_labels, test_size=0.5, stratify=test_labels if stratify is not None else None)

    return DatasetDict({
        'train': Dataset.from_dict({'Text': train_texts.tolist(), 'LabelID': train_labels.tolist()}),
        'validation': Dataset.from_dict({'Text': val_texts.tolist(), 'LabelID': val_labels.tolist()}),
        'test': Dataset.from_dict({'Text': test_texts.tolist(), 'LabelID': test_labels.tolist()})
    })


### model_training.py
import torch
import mlflow
import mlflow.transformers
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data_preprocessing import split_data
from config import *
import os

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
    def __init__(self, dataset):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(dataset['train']['LabelID'].unique()))

    def tokenize(self):
        return self.dataset.map(lambda x: self.tokenizer(x['Text'], truncation=True, padding=True, max_length=100), batched=True)

    def train(self):
        tokenized_data = self.tokenize()
        args = TrainingArguments(
            output_dir=f"./{MODEL_PATH_NAME}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            num_train_epochs=TRAIN_EPOCHS,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            fp16=torch.cuda.is_available()
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_data['train'],
            eval_dataset=tokenized_data['validation'],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if EARLY_STOPPING else []
        )
        mlflow.transformers.autolog()
        trainer.train()
        test_results = trainer.evaluate(tokenized_data['test'])
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
        return test_results


if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH).dropna()
    data = augment_data(data)
    dataset = split_data(data)
    trainer = NLPTrainer(dataset)
    test_results = trainer.train()
    print("Test Results:", test_results)