# E-BERT

E-BERT: Efficient-Yet-Effective Entity Embeddings for BERT (Poerner et al., 2020)

<https://github.com/npoe/ebert>

## Environment

```bash
conda create -n e-bert python=3.7.7
conda activate e-bert
pip install torch==1.2.0
cd code
cd kb
pip install -r requirements.txt
pip install overrides==2.8.0
python -c "import nltk; nltk.download('wordnet')"
python -m spacy download en_core_web_sm
pip install --editable .
pip install pytorch-transformers==1.2.0
pip install wikipedia2vec==1.0.4
pip install scikit-learn==0.21
cd ../..
bash prepare.sh
pip install pynif
```

Download the pretrained models (29.1 GiB):

```bash
wget https://www.cis.uni-muenchen.de/~poerner/blobs/e-bert/wikipedia2vec-base-cased
wget https://www.cis.uni-muenchen.de/~poerner/blobs/e-bert/wikipedia2vec-base-cased.bert-base-cased.linear.npy
mkdir mappers
mv wikipedia2vec-base-cased resources/wikipedia2vec
mv wikipedia2vec-base-cased.bert-base-cased.linear.npy mappers
```

Fit the linear mapping:

```bash
cd code
python3 run_mapping.py --src wikipedia2vec-base-cased --tgt bert-base-cased --save_out ../mappers/wikipedia2vec-base-cased.bert-base-cased.linear
```

Train a new model:

```bash
python run_aida.py --model_dir '../model'
```

This creates a model in the model folder that looks like:

```text
model/
├── model.pth
├── model_args.json
├── null_bias.pth
├── null_vector.pth
└── train_args.json
```

Score the new model:

```bash
python score_aida.py --pred '../model/aida_dev.txt.pred_iter3.txt' --gold '../model/aida_dev.txt.gold.txt'
```

## Evaluation

When starting the server, the following option is provided:

* model_dir: the folder containing the trained model. Typically `../model`

Use one of the following commands to start the annotator server.

Default:

```bash
python -m gerbil_connect.server_template --model_dir '../model'
```

Empty Candidate Sets:

```bash
python -m gerbil_connect.server_template --model_dir '../model' --no-candidate-sets
```

Full Candidate Sets:

```bash
python -m gerbil_connect.server_template --model_dir '../model' --full-candidate-sets
```

After starting the gerbil_connect server, use `http://localhost:3002/annotate_aida` in GERBIL as the annotator URL.

## Entity Linking

The following performs entity linking on the AIDA datasets without GERBIL. It can be easily changed to run on custom text.

```bash
python -m gerbil_connect.entity_linking --model_dir '../model' --no-candidate-sets
```
