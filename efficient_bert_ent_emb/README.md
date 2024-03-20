# Efficient Entity Embeddings for BERT

Efficient Entity Embedding Construction from Type Knowledge for BERT (Feng et al., 2022)

<https://github.com/yukunfeng/efficient_bert_ent_emb>

## Environment

```bash
conda create -n eee-bert python=3.7.7
conda activate eee-bert
pip install torch==1.2.0
cd entity_linking/code
cd kb
pip install -r requirements.txt
pip install overrides==2.8.0
python -c "import nltk; nltk.download('wordnet')"
python -m spacy download en_core_web_sm
pip install --editable .
pip install pytorch-transformers==1.2.0
pip install wikipedia2vec==1.0.4
pip install scikit-learn==0.21
pip install pynif
```

Download the entity types extracted through the instanceof relation from Wikidata from [here](https://drive.google.com/file/d/17ClfmuM65U_rRG4OHx34TMXvE0EEyiki/view?usp=sharing) (`wikidata_entity_types.tsv`, 513 MiB).

```bash
gzip -d  wikidata_entity_types.tsv.gz
mv wikidata_entity_types.tsv ./resources/
```

Download wikipedia2vec (29.1 GiB):

```bash
wget https://www.cis.uni-muenchen.de/~poerner/blobs/e-bert/wikipedia2vec-base-cased
mv wikipedia2vec-base-cased resources/wikipedia2vec
```

The directories should look like this:

```text
efficient_bert_ent_emb/
│
├── entity_linking/
│   │
│   ├── code/
│   │   ├── run_aida.py
│   │   └── model_dirs/
│   │       └── model/
│   │
│   └── data/
│       └── AIDA/
│           ├── aida_train.txt
│           ├── aida_dev.txt
│           └── aida_test.txt
│
└── resources/
    └── wikidata_entity_types.tsv
```

Train a new model:

```bash
python -u ./run_aida.py --model_dir "./model_dirs/model" --lr 3e-5 --epochs 4 --train_file ../data/AIDA/aida_train.txt --dev_file ../data/AIDA/aida_dev.txt --test_file ../data/AIDA/aida_test.txt --wikidata_entity_types_path ../../resources/wikidata_entity_types.tsv
```

This creates a model in the model folder that looks like:

```text
model/
├── ent2idx.pth
├── model.pth
├── model_args.json
├── null_bias.pth
├── null_vector.pth
└── train_args.json
```

Score the new model:

```bash
python score_aida.py --gold 'model_dirs/model/aida_dev.txt.gold.txt' --pred 'model_dirs/model/aida_dev.txt.pred.txt'
python score_aida.py --gold 'model_dirs/model/aida_test.txt.gold.txt' --pred 'model_dirs/model/aida_test.txt.pred.txt'
```

## Evaluation

When starting the server, the following option is provided:

* model_dir: the folder containing the trained model. Typically `./model_dirs/model`

Use one of the following commands to start the annotator server.

Default:

```bash
python -m gerbil_connect.server_template --model_dir './model_dirs/model'
```

Empty Candidate Sets:

```bash
python -m gerbil_connect.server_template --model_dir './model_dirs/model' --no-candidate-sets
```

Full Candidate Sets:

```bash
python -m gerbil_connect.server_template --model_dir './model_dirs/model' --full-candidate-sets
```

After starting the gerbil_connect server, use `http://localhost:3002/annotate_aida` in GERBIL as the annotator URL.

## Entity Linking

The following performs entity linking on the AIDA datasets without GERBIL. It can be easily changed to run on custom text.

```bash
python -m gerbil_connect.entity_linking --model_dir './model_dirs/model' --full-candidate-sets
```
