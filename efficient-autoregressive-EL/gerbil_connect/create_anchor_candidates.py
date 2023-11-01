from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

# use the ebert or eeebert conda environments.
from kb.wiki_linking_util import WikiCandidateMentionGenerator
candidate_generator = WikiCandidateMentionGenerator(entity_world_path = None, max_candidates = 1000)

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

import json

with open('AIDA_testc_gold.json', 'r', encoding='utf-8') as f:
    test_c = json.load(f)

test_c = test_c["annotations"]

aci_list = []

for item in test_c:
    text = item["text"]
    links = item["value"] # eg: {"start": 23, "end": 29, "tag": "http://en.wikipedia.org/wiki/London"}

    anchor_cand_item = {}
    anchor_cand_item["id"] = ""
    anchor_cand_item["input"] = text
    anchor_cand_item["anchors"] = []
    anchor_cand_item["candidates"] = []

    encoding = tokenizer(anchor_cand_item["input"], return_offsets_mapping=True)
    offset_mapping = encoding.offset_mapping

    for link in links:
        char_start = link["start"]
        char_end = link["end"]
        mention = anchor_cand_item["input"][char_start:char_end]
        gold_entity = link["tag"].replace("http://en.wikipedia.org/wiki/", "").replace("_", " ")

        token_start = next((token_id for token_id, (start, end) in enumerate(offset_mapping) if start <= char_start < end), None)
        token_end = next((token_id for token_id, (start, end) in enumerate(offset_mapping) if start < char_end <= end), None)

        if token_start is not None and token_end is not None:
            tokens = tokenizer.convert_ids_to_tokens(encoding.input_ids[token_start:token_end+1])
            print("Mention: ", mention)
            print("Tokens:", tokens)
        else:
            print("Character indices do not correspond to complete tokens in the text.")
            continue

        gold_mention = word_tokenize(mention)
        candidate_entities = candidate_generator.process(gold_mention) #eg: (entity_id, entity_candidate, p(entity_candidate | mention string))
        candidate_entities = [candidate[1].replace("_", " ") for candidate in candidate_entities]

        anchor_cand_item["anchors"].append([token_start, token_end, gold_entity])
        anchor_cand_item["candidates"].append(candidate_entities)
    
    aci_list.append(anchor_cand_item)

with open('aida_c_dataset.jsonl', 'w') as f:
    for entry in aci_list:
        json.dump(entry, f)
        f.write('\n')
