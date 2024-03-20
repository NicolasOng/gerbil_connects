# CHOLAN

CHOLAN: A modular approach for neural entity linking on Wikipedia and Wikidata (Kannan Ravi et al., 2021)

<https://github.com/ManojPrabhakar/CHOLAN>

## Environment

```bash
conda create --name cholan python=3.7
conda activate cholan
pip install torch torchvision torchaudio
pip install nltk transformers tensorflow==2.4.0 pandas keras==2.3.1 scikit-learn matplotlib seaborn natsort
```

## Downloads and Missing Files

[This link](https://figshare.com/articles/dataset/CHOLAN-EL-Dataset/13607282) contains an EL dataset (323 MiB) extracted from the [T-Rex dataset](https://hadyelsahar.github.io/t-rex/).

[This link](https://drive.google.com/open?id=1MfjzjZH_KKsXshtepzSBwkvjabdEytzh) contains the AIDA-CoNLL dataset (5.82 GiB) from the DCA paper, from [this](https://github.com/YoungXiyuan/DCA) repository.

Running `python cholan.py` in `CHOLAN/Cholan_CoNLL_AIDA/End2End/` performs and evaluates CHOLAN's end-to-end entity linking. However, this requires `ned_data.tsv`.

`ned_data.tsv` seems to be created by `NED_Dataset_Creation.py`. However, this script requires `ner_data.tsv`. We are not sure where this file is created, and it is not in any of the downloads.

Additionally, the trained NED and NER models required by `cholan.py` aren't available and there is no training code in the repository. The model directories are `/data/prabhakar/CG/NED_pretrained/model_data_50000/` and `/data/prabhakar/manoj/code/NER/BERT-NER-CoNLL/pretrained_ner/` in the file.

Due to this, we were unable to replicate CHOLAN.
