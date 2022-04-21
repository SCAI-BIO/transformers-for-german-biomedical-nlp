# Critical Assessment of Transformer-based AI Models for German Clinical Notes

This repository provides the code for the publication **Critical Assessment of Transformer-based AI Models for German
Clinical Notes**. In that study, we assessed the performance of eight published general-purpose and three newly trained
transformer-based language models specific to the biomedical domain.

## Repository structure

```text
.
├── fine-tuning                             # Directory which contains all scripts for fine-tuning
│   ├── bronco
│   │   └── data
│   │       ├── config
│   │       ├── databases
│   │       ├── processed
│   │       └── raw
│   ├── chadl
│   │   ├── data
│   │   │   ├── config
│   │   │   ├── interim
│   │   │   └── raw
│   │   └── helpers
│   ├── ggponc
│   │   └── data
│   │       ├── config
│   │       ├── processed
│   │       └── raw
│   └── jsyncc
│       └── data
│           └── config
├── pre-training                            # Directory which contains the scripts to pre-train BioGottBERT and BioELECTRA
└── toolbox                                 # Python module with the code used in the pre-training and fine-tuning scripts
    ├── datasets
    ├── models
    ├── pretraining
    ├── training
    │   ├── ner
    │   └── sequence_classification
    └── utils

```

## Data availability

Unfortunately, the pre-training and the ChaDL datasets are not publicly available. However, the BRONCO150, GGPONC, and
JSynCC dataset can be acquired. The [BRONCO150](https://www2.informatik.hu-berlin.de/~leser/bronco/index.html)
and [GGPONC](https://www.leitlinienprogramm-onkologie.de/projekte/ggponc-english/) datasets can be requested by the
authors directly, and [the JSynCC repository](https://github.com/JULIELab/jsyncc) provides code to generate the JSynCC
dataset.

## Dependencies

## Sharing the pre-trained models

We made our domain-adapted BioGottBERT model available in the Huggingface model hub
at https://huggingface.co/SCAI-BIO/bio-gottbert-base.

## Usage

### Pre-training

The following three commands are needed to pre-train BioELECTRA models.

```bash
python 0_create_custom_tokenizer.py custom_tokenizer -i test.txt -i test2.txt --vocab-size=15000
python 1_create_pretraining_data.py 512 example -i data/raw/example.txt -v data/processed/vocab.txt --threads 16 --model ELECTRA
python 2_pretrain.py electra-small BioELECTRA data/processed/ELECTRA_example-512_training.pt data/processed/ELECTRA_example-512_validation.pt  data/processed/vocab.txt
```

The following two commands are needed to run the domain-adaption script:

```bash
python 1_create_pretraining_data.py 512 example -i data/raw/example.txt -t ~/git/german-clinical-bert/model_evaluation/bronco/data/external/gottbert-base --threads 16 --model RoBERTa
python 2_pretrain.py domain-adapted-roberta roberta_output data/processed/RoBERTa_example-512_training.pt data/processed/RoBERTa_example-512_validation.pt uklfr/gottbert-base
```

### Fine-tuning

The relevant files for fine-tuning can be found in the fine-tuning subdirectories. Initially, the data must be
preprocessed. For this purpose, a 0_generate_dataset.py file is available in each folder. Subsequently, the
1_train_model.py or 1_run_ncv.py files must be executed specifying a configuration file for each model and, if
necessary, for each outer fold.

## Contact

Please post a GitHub issue if you have any questions.