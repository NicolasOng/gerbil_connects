# End-to-End Neural Entity Linking

End-to-End Neural Entity Linking (Kolitsas et al., 2018)

<https://github.com/dalab/end2end_neural_el>

## Environment

```bash
conda create --name e2e_env python=3.6
conda activate e2e_env
conda install -c conda-forge tensorflow=1.4.0
conda install enum34 requests scipy tensorflow-tensorboard numpy termcolor gensim nltk
python -c "import nltk; nltk.download('punkt')"
pip install flask flask-cors pynif
```

Download the 'data' folder from [this link](https://drive.google.com/file/d/1OSKvIiXHVVaWUhQ1-fpvePTBQfgMT6Ps/view?usp=sharing) (5.82 GiB), unzip it and place it under end2end_neural_el/

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

The following performs entity linking on the AIDA datasets without GERBIL. It can be easily to run on custom text.

```bash
python -m gerbil_connect.entity_linking
```
