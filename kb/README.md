# KnowBert

Knowledge Enhanced Contextual Word Representations (Peters et al., 2019)

<https://github.com/allenai/kb>

## Environment

```bash
cd gerbil_connects/kb
conda create -n knowbert python=3.6.7
conda activate knowbert
pip install torch==1.2.0
pip install -r requirements.txt
pip install overrides==2.8.0
python -c "import nltk; nltk.download('wordnet')"
python -m spacy download en_core_web_sm
pip install --editable .
pip install scikit-learn==0.21
pip install flask flask_cors pynif nltk
```

Make sure the tests pass:

```bash
pytest -v tests
```

Download the pretrained models (2.91 GiB):

* [KnowBert-Wiki](https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_model.tar.gz)
* [KnowBert-W+W](https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_wordnet_model.tar.gz)

## Evaluation

When starting the server, the following options are provided:

* model_archive: the folder containing the pretrained model. Either `/.../knowbert_wiki_wordnet_model` or `/.../knowbert_wiki_model`.
* wiki_and_wordnet: use this flag if using the wiki_wordnet model

Use one of the following commands to start the annotator server.

Default:

```bash
python -m gerbil_connect.server_template --model_archive '/.../knowbert_wiki_wordnet_model' --wiki_and_wordnet
```

Empty Candidate Sets:

```bash
python -m gerbil_connect.server_template --model_archive '/.../knowbert_wiki_model' --no-candidate-sets
```

Full Candidate Sets:

```bash
python -m gerbil_connect.server_template --model_archive '/.../knowbert_wiki_model' --full-candidate-sets
```

After starting the gerbil_connect server, use `http://localhost:3002/annotate_aida` in GERBIL as the annotator URL.

## Entity Linking

The following performs entity linking on the AIDA datasets without GERBIL. It can be easily changed to run on custom text.

```bash
python -m gerbil_connect.entity_linking --model_archive '/.../knowbert_wiki_wordnet_model' --wiki_and_wordnet --full-candidate-sets
```
