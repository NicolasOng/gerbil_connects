# Entity Knowledge in BERT

Investigating Entity Knowledge in BERT with Simple Neural End-To-End Entity Linking (Broscheit, 2019)

<https://github.com/samuelbroscheit/entity_knowledge_in_bert>

## Environment

```bash
conda create -n broscheit python=3.8
conda activate broscheit
pip install -r requirements.txt
git submodule update --init
source setup_paths
pip install requests tqdm regex

pip install torch torchvision torchaudio
pip install numpy==1.20 pandas==1.2.1
```

Get the AIDA-CoNLL benchmark file (5.21 MiB) from [here](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/aida/downloads/) and place it in `data/benchmarks/aida-yago2-dataset/AIDA-YAGO2-dataset.tsv`.

## Training the Model

Preprocessing the data

```bash
python bert_entity/preprocess_all.py -c config/conll2019__preprocess.yaml
```

Pretraining on Wikipedia

```bash
python bert_entity/train.py -c config/conll2019__train_on_wiki.yaml
```

Finetuning on AIDA-CoNLL Benchmark

```bash
python bert_entity/train.py -c config/conll2019__train_on_aida_conll.yaml
```

Evaluating the model

```bash
python bert_entity/train.py -c config/conll2019__train_on_aida_conll.yaml --eval_on_test_only True --resume_from_checkpoint data/checkpoints/conll2019_aidaconll_00001/best_f1-0.pt
```

While we were able to run the training scripts, our trained model did not match the performance reported in the paper (lower by ~15%). The time needed to retrain and integrate the model into our evaluation environment exceeds the scope of our paper.
