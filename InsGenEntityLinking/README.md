# Insructed Generative Entity Linking

Instructed Language Models with Retriever Are Powerful Entity Linkers (Xiao et al., 2023)

<https://github.com/MrZilinXiao/InsGenEntityLinking>

## Environment

```shell
conda create -n xiao python=3.8.10
conda activate xiao

pip install wikipedia2vec
export PATH="/home/nicolasong/.local/bin:$PATH"

pip install transformers==4.27.4 tokenizers==0.13.3 sentencepiece
pip install torch torchvision torchaudio pyjnius deepspeed peft wandb

export CLASSPATH=$CLASSPATH:resources/apache-opennlp-1.5.3/lib/*
export PYTHONPATH="${PYTHONPATH}:./src"
```

Download the latest Wikipedia English dump (20.2GiB) from [here](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2).

Download opennlp-1.5.3 (20 MiB) from [here](https://archive.apache.org/dist/opennlp/opennlp-1.5.3/), and place it in the repo to create `InsGenEntityLinking/src/common_utils/resources/apache-opennlp-1.5.3/lib/*`.

Download `en-sent.bin` (96.2 KiB) from [here](https://opennlp.sourceforge.net/models-1.5/), and place it in the `resources/` directory.

Download Llama-7B converted to work with Transformers/HuggingFace via HF's conversion script (25.1 GiB).

```shell
git lfs install
git clone https://huggingface.co/luodian/llama-7b-hf
```

## Training the Model

Use `wikipedia2vec` to build "dump db" from the Wikipedia dump.

```shell
wikipedia2vec build-dump-db enwiki-latest-pages-articles.xml.bz2 enwiki-latest-pages-articles.db
```

Convert the dump db to the jsonl input-output format of InsGenEL.

```shell
python data_scripts/wiki_dump.py
```

Train the model.

```shell
deepspeed universal_train.py --model_name_or_path ./llama-7b-hf --train_jsonl_path ./0220-full-para-filtered.jsonl --empty_instruction False --bf16 False --output_dir ./llama_7B_final_checkpoint/ --num_train_epochs 1 --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 1 --save_strategy steps --save_steps 50000 --save_total_limit 10 --learning_rate 2e-5 --weight_decay 0. --lr_scheduler_type polynomial --warmup_ratio 0.03 --logging_steps 10 --deepspeed ./deepspeed_configs/ds_config_zero3.json --fp16 --report_to wandb --run_name llama_7B_final_checkpoint --tf32 False
```

We were unable to get this script to finish without errors in time for the paper.
