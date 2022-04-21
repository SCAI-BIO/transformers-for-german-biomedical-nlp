# -*- coding: utf-8 -*-
import configparser

config = configparser.ConfigParser()
config["Data"] = {
    "whole_data": "data/processed/whole_dataset.pt",
    "training_data": "data/processed/training_dataset.pt",
    "validation_data": "data/processed/validation_dataset.pt",
    "testing_data": "data/processed/testing_dataset.pt",
    "label_set": "data/processed/label_set.pt",
}
config["Model"] = {
    "name": "gottbert-base",
    "model": "uklfr/gottbert-base",
    "tokenizer": "data/external/gottbert-base",
    "tokenizer_type": "pretrained",
    "model_function": "RobertaForTokenClassification",
    "tokenizer_function": "RobertaTokenizerFast",
    "registered_model": "",
}
config["Training"] = {
    "learning_rate": 2e-5,
    "epochs": 60,
    "warmup_steps": 50,
    "eval_batch_size": 16,
    "train_batch_size": 16,
    "early_stopping_patience": 30,
    "early_stopping_threshold": 0.1,
    "metric": "eval_macro_f1",
    "greater_is_better": True,
    "trainer_function": "Trainer",
    "trainer_module": "transformers",
}
config["Logging"] = {
    "experiment_name": "chadl",
    "tracking_uri": "http://localhost:45823",
}
config["Storage"] = {"results": "reports/tables"}
config["Ganbert"] = {"use": False}
with open("default.ini", "w") as configfile:
    config.write(configfile)
