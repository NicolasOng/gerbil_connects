Pytorch Implementation of AACL Findings 2022 paper ["Efficient Entity Embedding
Construction from Type Knowledge for BERT"](to_add)

## Download entity types
We extracted entity types through instanceof relation from Wikidata and it can be downloaded from [here](https://drive.google.com/file/d/17ClfmuM65U_rRG4OHx34TMXvE0EEyiki/view?usp=sharing) (around 140MB).
After downloading, unzip it and move to `./resources/`:
```
gzip -d  wikidata_entity_types.tsv.gz
mv wikidata_entity_types.tsv ./resources/
```

Each line in `wikidata_entity_types.tsv` is as follows:
```
entity_id[TAB]entity_label[TAB]entity_wikipedia_title[TAB]entity_ids_connected_by_instanceof
```

## Entity linking experiment
Our code for entity linking experiment is adapted from [E-BERT](https://github.com/NPoe/ebert).
To set the environment for E-BERT, use the following command:

```
cd entity_linking
conda env create -f environment.yaml
conda activate e-bert
```

Then download wikipedia2vec (around 30GB, used only to get normalized wikipedia titles):
```
wget https://www.cis.uni-muenchen.de/~poerner/blobs/e-bert/wikipedia2vec-base-cased
mv wikipedia2vec-base-cased resources/wikipedia2vec
```

Finally, install [KnowBert](https://github.com/allenai/kb) for candidate entity generation.
```
cd code/kb
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"
python -m spacy download en_core_web_sm
pip install --editable .
```

Use following command to train model on AIDA dataset and the score on test dataset will be reported.

```
python -u ./run_aida.py \
    --model_dir "./model_dirs/model" \
    --lr 3e-5 \
    --epochs 4 \
    --train_file ../data/AIDA/aida_train.txt \
    --dev_file ../data/AIDA/aida_dev.txt \
    --test_file ../data/AIDA/aida_test.txt \
    --wikidata_entity_types_path ../../resources/wikidata_entity_types.tsv
```

## Relation classification and entity typing experiments
Our code for the two experiments are adapted from
[ERNIE](https://github.com/thunlp/ERNIE) and the requirements are as follows:
- Python version >= 3.6.5
- Pytorch version 1.2.0
- tqdm
- boto3
- requests

Use following commands to download Bertbase:
```
cd bert_base
wget https://huggingface.co/bert-base-uncased/raw/2f07d813ca87c8c709147704c87210359ccf2309/vocab.txt
wget https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
```

### Relation classification
First, generate entity embedding, computed through its types, for fewrel dataset:
```
python ./code/descrip_emb_util.py --data_dir data/fewrel \
    --ernie_model bert_base \
    --entities_tsv ../resources/wikidata_entity_types.tsv \
    --do_lower_case \
    --output_base fewrel_type_emb_out \
    --max_seq_length 10
```

The, run following command to train on fewrel and the score on test
dataset will be reported.

```
python3 code/run_fewrel.py \
    --data_dir ./data/fewrel --ernie_model bert_base \
    --emb_base ./fewrel_type_emb_out --do_lower_case \
    --max_seq_length 256 --train_batch_size 16 --learning_rate 4e-5 \
    --num_train_epochs 10 --output_dir "fewrel_output_dir" \
    --entities_tsv ../resources/wikidata_entity_types.tsv
```

### Entity typing
Similar as above, generate entity embedding for openentity dataset:

```
python ./code/descrip_emb_util.py --data_dir data/OpenEntity \
    --ernie_model bert_base \
    --entities_tsv ../resources/wikidata_entity_types.tsv \
    --do_lower_case \
    --output_base ./openentity_type_emb_out \
    --max_seq_length 10
```

And then train and report the score on test dataset.

```
python code/run_openentity.py \
  --emb_base openentity_type_emb_out \
  --do_train --do_lower_case \
  --data_dir ./data/OpenEntity --ernie_model bert_base \
  --max_seq_length 256 --train_batch_size 16 --learning_rate 2e-5 \
  --num_train_epochs 10.0 --output_dir openentity_out_dir \
  --entities_tsv ../resources/wikidata_entity_types.tsv
```
