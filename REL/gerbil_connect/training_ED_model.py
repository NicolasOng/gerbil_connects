# https://rel.readthedocs.io/en/latest/tutorials/deploy_REL_new_wiki/#training-your-own-entity-disambiguation-model

from REL.training_datasets import TrainingEvaluationDatasets
from REL.entity_disambiguation import EntityDisambiguation

wiki_version = "wiki_2019"
base_url = "/mnt/d/Datasets/Radboud/"

datasets = TrainingEvaluationDatasets(base_url, wiki_version).load()

config = {
    "mode": "train",
    "model_path": "{}/{}/generated/model".format(
        base_url, wiki_version
    ),
}

# Add the parameter if wiki_version is 2019
if wiki_version == "wiki_2019":
    config["dev_f1_change_lr"] = 0.88

model = EntityDisambiguation(base_url, wiki_version, config)

# 3. Train or evaluate model
if config["mode"] == "train":
    model.train(
        datasets["aida_train"], {k: v for k, v in datasets.items() if k != "aida_train"}
    )
else:
    model.evaluate({k: v for k, v in datasets.items() if "train" not in k})

# 3. Train and predict using LR

model_path_lr = "{}/{}/generated/".format(base_url, wiki_version)

model.train_LR(
    datasets,
    model_path_lr
)
