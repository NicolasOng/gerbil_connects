# Radboud Entity Linker

REL: An Entity Linker Standing on the Shoulders of Giants (van Hulst
et al., 2020)

<https://github.com/informagi/REL.git>

## Environment

```bash
conda create -n rel
conda activate rel
conda install pip
pip install -e .
pip install flask flask_cors pynif
```

Download the wiki 2014 and 2019 data from here (99.4 GiB): <https://rel.readthedocs.io/en/latest/#download>

Put the data into a folder with a directory structure like:

```text
Radboud/
├── ed-wiki-2014
├── ed-wiki-2019
├── generic
├── wiki_2014
└── wiki_2019
```

## Evaluation

Use one of the following commands to start the annotator server.
Use the following options:

- base: the folder containing the downloaded data
- wiki: decides which of the downloaded data to use. 2014 or 2019

Default:

```bash
python -m gerbil_connect.server_template --wiki 2019 --base /home/Radboud/
```

Empty Candidate Sets:

```bash
python -m gerbil_connect.server_template --no-candidate-sets --wiki 2019 --base /home/Radboud/
```

Full Candidate Sets:

```bash
python -m gerbil_connect.server_template --full-candidate-sets --wiki 2019 --base /home/Radboud/
```

After starting the gerbil_connect server, use `http://localhost:3002/annotate_aida` in GERBIL as the annotator URL.

## Entity Linking

The following performs entity linking on the AIDA datasets without GERBIL. It can be easily changed to run on custom text.

```bash
python -m gerbil_connect.entity_linking --wiki 2019 --base /home/Radboud/
```
