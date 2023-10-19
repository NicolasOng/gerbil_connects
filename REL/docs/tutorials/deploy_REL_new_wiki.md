# Deploy REL for a new Wikipedia corpus

Although we will do our best to continuously provide the community with recent corpuses, it may be the case that a user wants to, for example, use
an older corpus for a specific evaluation. For this reason we provide the user with the option to do so. We must, however,
note that some steps are outside the scope of the REL package, which makes support for some of these steps a difficult task.

This tutorial is divided in four parts. The first part deals with [extracting a Wikipedia corpus and creating a p(e|m) index](#extracting-a-wikipedia-dump).
After extracting the aforementioned index and thus obtaining a sqlite3 database, we are also in need of Embeddings. We obtain new embeddings by [training a Wikipedia2Vec model](#training-wikipedia2vec-embeddings).
To train our own Entity Disambiguation model, we need to [generate training, validation and test files](#generate-training-validation-and-test-files).
These aforementioned p(e|m) index,  can be used to [train your own Entity Disambiguation model](#training-your-own-entity-disambiguation-model). 
After obtaining this model, a user may choose to [evaluate the obtained model on Gerbil](../evaluate_gerbil/) or for [E2E Entity Linking](../e2e_entity_linking/).

## Creating a folder structure

Previously we have defined our `base_url`, but now we need to create a new sub folder in our directory to obtain 
the following folder structure:

```
.
├── generic
└─── wiki_2014
|   ├── basic_data
|      └── anchor_files
|   └── generated
└─── wiki_2019
|   ├── basic_data
|      └── anchor_files
|   └── generated
└─── your_corpus_name
|   ├── basic_data
|      └── anchor_files
|   └── generated
```

### Extracting a Wikipedia dump

There are several platforms that host [Wikipedia dumps](https://dumps.wikimedia.org/). These platforms provide `xml` files that need processing. 
A tool that does this is called [WikiExtractor](https://github.com/attardi/wikiextractor). This tool takes as an input a
Wikipedia dump and spits out files that are required for our package. We, however, had to alter it slightly such that it 
stored some additional files that are required for this package. As such, we have added this edited edition to our scripts
folder. To process a Wikipedia dump run the command below in a terminal. We define the file size (`bytes`) as one GB, but it can
be changed based on the user's wishes. We advice users to run the script in the `basic_data` folder. After the script is
done, the user only needs to copy the respective wikipedia dump (this excludes the wiki id/name mapping, disambiguation pages
and redirects) into the `anchor_files` folder. 

```
python WikiExtractor.py ./wiki_corpus.xml --links --filter_disambig_pages --processes 1 --bytes 1G
```

### Generate p(e\|m) index

Now that we have extracted the necessary data from our Wikipedia corpus, we may create the p(e|m) index. This index
is automatically stored in the same database as the embeddings that can be found in the `generated` folder. The first
thing we need to do is define define a variable where we store our database. Secondly, we instantiate a `Wikipedia` class
that loads the wikipedia id/name mapping, disambiguation file and redirect file. 

```python
wiki_version = "your_corpus_name"
wikipedia = Wikipedia(base_url, wiki_version)
```

Now all that is left is to instantiate our `WikipediaYagoFreq` class that is responsible for parsing the Wikipedia and
YAGO articles. Here we note that the function `compute_custom()` by default computes the p(e|m) probabilities of
YAGO, but that it can be replaced by any Knowledge Base of your choosing. To replace YAGO, make sure that the input dictionary
to the aforementioned function is of the following format: 
`{mention_0: {entity_0: cnt_0, entity_n: cnt_n}, ... mention_n: {entity_0: cnt_0, entity_n: cnt_n}}`

```python
wiki_yago_freq = WikipediaYagoFreq(base_url, wiki_version, wikipedia)
wiki_yago_freq.compute_wiki()
wiki_yago_freq.compute_custom()
wiki_yago_freq.store()
```

## Embeddings

### Training Wikipedia2Vec embeddings
Training new embeddings is based on the [Wikipedia2Vec](http://wikipedia2vec.github.io/) package. For extra information
about this package we refer to their website. We did, however, feel obligated to provide users with the same scripts that
we used to train our embeddings. These two shell scripts first install Wikipedia2vec and then asks you where
your Wikipedia dump is stored. Please make sure that the dump is still zipped and thus has the extensions `.xml.bz2`.
The two scripts are located in `scripts/w2v`, where you first run `preprocess.sh` which requires you to enter the
location of your Wikipedia dump. After this is done, you can run `train.sh` which will train a Wikipedia2Vec model and
store it in the required word2vec format.

### Storing Embeddings in DB

Now that the Embeddings are trained and stored you might notice that the file is huge. This is exactly the reason
why we choose for a database approach, because it was simply infeasible to load all the embeddings into memory. After 
the package is installed, all we have to do is run the code below. Please make sure to not not change the variables `save_dir`
and `db_name`. The variable `embedding_file` needs to point to the trained Wikipedia2vec file.

```python
from REL.db.generic import GenericLookup

save_dir = "{}/{}/generated/".format(base_url, wiki_version)
db_name = "entity_word_embedding"
embedding_file = "./enwiki_w2v_model"

# Embedding load.
emb = GenericLookup(db_name, save_dir=save_dir, table_name='embeddings')
emb.load_word2emb(embedding_file, batch_size=5000, reset=True)
```

## Generating training, validation and test files

To train your own Entity Disambiguation model, training, validation and test files are required. To obtain these
files we first define our `wiki_version` and instantiate a `Wikipedia` class that are required.

```python
from REL.wikipedia import Wikipedia
from REL.generate_train_test import GenTrainingTest

wiki_version = "wiki_2019/"
wikipedia = Wikipedia(base_url, wiki_version)
```
Secondly, we instantiate the class `GenTrainingTest` that parses the raw training and test files that can be found in our
`generic` folder. The user may choose to only retrieve one of the listed datasets below by simply changing the name
in the function `process_wned` or `process_aida`. The reason for separating these functions was due to the way they were
provided and had to be parsed.

```python
data_handler = GenTrainingTest(base_url, wiki_version, wikipedia)
for ds in ["aquaint", "msnbc", "ace2004", "wikipedia", "clueweb"]:
    data_handler.process_wned(ds)

for ds in ["train", "test"]:
    data_handler.process_aida(ds)
```

## Training your own Entity Disambiguation model

For our final tutorial we will train our own Entity Disambiguation model. The first step, as always, is to import
the necessary packages and to define the `wiki_version`.

```python
from REL.training_datasets import TrainingEvaluationDatasets
from REL.entity_disambiguation import EntityDisambiguation

wiki_version = "wiki_2019"
```

Our second step is loading our training and evaluation datasets that can be found in our `generated` folder. These
datasets are provided by us, but can also be generated by the user in previous steps.

```python
datasets = TrainingEvaluationDatasets(base_url, wiki_version).load()
```


Our third step consists of defining the config dictionary that serves as an input to the model. If the mode is set 
to `eval` then the model will evaluate the given datasets given that the model exists (else it will throw an error). The 
mode should be set to `train` if the user wishes to train a new model and save it in the given `model_path`.

```python
config = {
    "mode": "train",
    "model_path": "{}/{}/generated/model".format(
        base_url, wiki_version
    ),
}
model = EntityDisambiguation(base_url, wiki_version, config)
```

Our fourth step consists of training or evaluating the model. This is dependent on the `mode` that was chosen previously,

```python
# 3. Train or evaluate model
if config["mode"] == "train":
    model.train(
        datasets["aida_train"], {k: v for k, v in datasets.items() if k != "aida_train"}
    )
else:
    model.evaluate({k: v for k, v in datasets.items() if "train" not in k})
```

Now that we have obtained our model, we want to express how confident we are in our prediction.
By default this is not integrated in our model, but we implemented an additional step, where we use Logistic Regression
to obtain confidence scores. However, to obtain such scores, we first need to train the model. This can be done by
running the code below.

```python
# 3. Train and predict using LR

model_path_lr = "{}/{}/generated/".format(base_url, wiki_version)

model.train_LR(
    datasets,
    model_path_lr
)
```