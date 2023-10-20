from REL.mention_detection import MentionDetection
from REL.utils import process_results
from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import Cmns, load_flair_ner

wiki_version = "wiki_2019"
base_url = "/mnt/d/Datasets/Radboud/"

# set up the MD module
mention_detection = MentionDetection(base_url, wiki_version)
tagger_ner = load_flair_ner("ner-fast") # they use this for the results in their paper.
#tagger_ngram = Cmns(base_url, wiki_version, n=5)

#set up the ED module
if wiki_version == "wiki_2014":
    model_path = "ed-wiki-2014"
elif wiki_version == "wiki_2019":
    model_path = "ed-wiki-2019"
config = {
    "mode": "eval",
    "model_path": model_path,
}
Radboud_model = EntityDisambiguation(base_url, wiki_version, config)

def entity_linking(raw_text):
    # process the raw text into the input format
    doc_name = "my_doc"
    input_text = {
        doc_name: (raw_text, []),
    }

    # mention detection using flair
    mentions, n_mentions = mention_detection.find_mentions(input_text, tagger_ner)

    # entity disambiguation
    predictions, timing = Radboud_model.predict(mentions)
    
    result = process_results(mentions, predictions, input_text)
    print(result)
    # process results
    '''
    Example result format:
    {'my_doc': [(0, 13, 'Hello, world!', 'Hello_world_program', 0.6534378618767961, 182, '#NGRAM#')]}
    '''
    mention_list = result.get(doc_name, [])
    output = [(mention[0], mention[0] + mention[1], mention[3]) for mention in mention_list]
    return output

def print_results(raw_text, output):
    for start, end, entity in output:
        print(start, end, entity)
        print(raw_text[start:end])
        print(raw_text[start-10:start] + "*" + raw_text[start:end] + "*" + raw_text[end:end+10])
        print("")

test_text = "Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire, on 14 March 1879. He later moved to Switzerland where he attended the Swiss Federal Institute of Technology. In 1905, Einstein published four groundbreaking papers in the journal Annalen der Physik, which are now known as the Annus Mirabilis papers."
output = entity_linking(test_text)
print_results(test_text, output)
