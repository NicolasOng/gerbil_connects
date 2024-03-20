# GENRE

Autoregressive entity retrieval (De Cao et al., 2021b)

<https://github.com/facebookresearch/GENRE>

<https://github.com/ad-freiburg/GENRE>

## Environment

```bash
conda create --name genre python=3.7
conda activate genre

git clone https://github.com/facebookresearch/KILT.git
cd KILT
pip install -e .
cd ..

git clone --branch fixing_prefix_allowed_tokens_fn https://github.com/nicola-decao/fairseq
cd fairseq
pip install --editable ./
python setup.py build develop
cd ..

conda install pytorch=1.6 torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c huggingface transformers=4.2.0

pip install unidecode requests bs4 marisa-trie pytest flask flask_cors pynif nltk spacy
python3 -m spacy download en_core_web_sm
pip install numpy==1.19.5
```

Download the `mention_to_candidates_dict.pkl` and `mention_trie.pkl` (3.1 GiB):

```bash
wget https://ad-research.cs.uni-freiburg.de/data/GENRE/data.zip
unzip data.zip -d data
```

Download the model (1.89 GiB):

```bash
wget http://dl.fbaipublicfiles.com/GENRE/fairseq_e2e_entity_linking_aidayago.tar.gz -P models
tar -xzvf models/fairseq_e2e_entity_linking_aidayago.tar.gz -C models
rm models/fairseq_e2e_entity_linking_aidayago.tar.gz
```

Update `GENRE/gerbil_connect/config.json`:

```json
{
    "MODEL_LOCATION": "/.../GENRE/models/fairseq_e2e_entity_linking_aidayago",
    "GENRE_REPO_LOCATION": "/.../gerbil_connects/GENRE/",
    "FAIRSEQ_REPO_LOCATION": "/.../fairseq/"
  }
```

## Evaluation

Use one of the following commands to start the annotator server.

Default:

```bash
python -m gerbil_connect.server_template
```

Empty Candidate Sets:

```bash
python -m gerbil_connect.server_template --no-candidate-sets
```

Full Candidate Sets:

```bash
python -m gerbil_connect.server_template --full-candidate-sets
```

After starting the gerbil_connect server, use `http://localhost:3002/annotate_aida` in GERBIL as the annotator URL.

## Entity Linking

The following performs entity linking on the AIDA datasets without GERBIL. It can be easily changed to run on custom text.

```bash
python -m gerbil_connect.entity_linking
```
