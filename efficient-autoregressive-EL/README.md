# Efficient Autoregressive Entity Linking

Highly Parallel Autoregressive Entity Linking with Discriminative Correction (De Cao et al., 2021a)

<https://github.com/nicola-decao/efficient-autoregressive-EL>

## Environment

```bash
conda create --name ea-el python=3.8
conda activate ea-el

pip install torch torchvision torchaudio transformers jsonlines pytorch_lightning==1.3.0 torchmetrics==0.6.0 ipython
pip install setuptools==59.5.0

pip install flask flask_cors pynif nltk
```

Download everything from [here](https://mega.nz/folder/l4RhnIxL#_oYvidq2qyDIw1sT-KeMQA) (1.30 GiB). This includes the model and pre-processed data needed to run the model on the AIDA dataset.

Update `efficient-autoregressive-EL/gerbil_connect/config.json`:

```json
{
    "REPO_LOCATION": "/.../gerbil_connects/efficient-autoregressive-EL/"
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

The following performs entity linking on the AIDA datasets without GERBIL. It can be easily to run on custom text.

```bash
python -m gerbil_connect.
entity_linking
```
