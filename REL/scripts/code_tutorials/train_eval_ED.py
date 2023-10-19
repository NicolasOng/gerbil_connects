from REL.entity_disambiguation import EntityDisambiguation
from REL.training_datasets import TrainingEvaluationDatasets

base_url = ""
wiki_version = "wiki_2014"

# 1. Load datasets # '/mnt/c/Users/mickv/Google Drive/projects/entity_tagging/deep-ed/data/wiki_2019/'
datasets = TrainingEvaluationDatasets(base_url, wiki_version).load()

# 2. Init model, where user can set his/her own config that will overwrite the default config.
config = {
    "mode": "eval",
    "model_path": "{}/{}/generated/model".format(base_url, wiki_version),
}
model = EntityDisambiguation(base_url, wiki_version, config)

# 3. Train or evaluate model.
if config["mode"] == "train":
    model.train(
        datasets["aida_train"], {k: v for k, v in datasets.items() if k != "aida_train"}
    )
else:
    model.evaluate({k: v for k, v in datasets.items() if "train" not in k})
