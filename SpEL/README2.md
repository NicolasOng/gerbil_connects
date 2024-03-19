# SpEL

SpEL: Structured Prediction for Entity Linking (Shavarani and Sarkar, 2023)

<https://github.com/shavarani/SpEL>

## Environment

```bash
conda create --name spel python=3.9
conda activate spel
pip install -r requirements.txt
pip install flask flask-cors pynif

export PYTHONPATH=/.../gerbil_connects/SpEL/src
cd /.../SpEL/src/spel
```

## Evaluation

When starting the server, the following options are provided:

* n, k, or pg - which candidate set to use.
  * n = none
  * k = kb+yago
  * pg = PPRforNED
* default, empty, full - candidate ablation set setting
  * default - use the chosen candidate set
  * empty - replace the candidate set with an empty one
  * full - replace the candidate set with a full one

Here is the command to start the server:

```bash
python server.py spel [n, k, pg] [default, empty, full] 500K
```

These are the commands to repeat our experiments:

Default:

```bash
python server.py spel n default 500K
python server.py spel k default 500K
python server.py spel k default 500K
```

Without Candidate Set:

```bash
python server.py spel n default 500K
```

After starting the gerbil_connect server, use `http://localhost:3002/annotate_aida` in GERBIL as the annotator URL.
