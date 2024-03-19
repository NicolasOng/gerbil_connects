# EntQA

EntQA: Entity Linking as Question Answering (Zhang et al., 2022)

<https://github.com/WenzhengZhang/EntQA>

## Environment

```bash
conda create --name entqa python=3.8
conda activate entqa

pip install -r requirements.txt
pip install faiss-gpu transformers==4.0.0
```

Download all the preprocessed data [here](https://drive.google.com/drive/folders/1DQvfjKOuOoUE3YcYrg2GIvODaOEZXMdH?usp=sharing) (6.62 GiB).

Download their trained retreiver model [here](https://drive.google.com/file/d/1bHS5rxGbHJ5omQ-t8rjQogw7QJq-qYFO/view?usp=sharing) (7.49 GiB).

Download their cached entity embeddings [here](https://drive.google.com/file/d/1znMYd5HS80XpLpvpp_dFkQMbJiaFsQIn/view?usp=sharing) (22.5 GiB).

Download their trained reader [here](https://drive.google.com/file/d/1A4I1fJZKxmROIE1fd0mdXN6b1emP_xt4/view?usp=sharing) (1.24 GiB).

Download the BLINK pretrained retreiver model (29.4 GiB).

```bash
git clone https://github.com/facebookresearch/BLINK.git
cd BLINK
chmod +x download_blink_models.sh
./download_blink_models.sh
cd ..
```

## Evaluation

Use the following command to start the annotator server. EntQA does not use pre-computed candidate sets when performing entity linking.

```bash
python -m gerbil_connect.server_template --add_topic --do_rerank
```

After starting the gerbil_connect server, use `http://localhost:3002/annotate_aida` in GERBIL as the annotator URL.
