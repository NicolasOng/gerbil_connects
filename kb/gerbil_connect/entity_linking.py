from kb.include_all import TokenizerAndCandidateGenerator

from allennlp.data import DatasetReader, Vocabulary, DataIterator
from allennlp.common import Params

from allennlp.nn.util import move_to_device

from allennlp.models.archival import load_archive

from allennlp.data import Instance
from allennlp.data.dataset import Batch

from kb.entity_linking import EntityLinkingBase

import torch

import pickle
import inspect

reader_params_ww = Params({
        "type": "aida_wiki_linking",
        "entity_disambiguation_only": False,
        "entity_indexer": {
            "type": "characters_tokenizer",
            "namespace": "entity_wiki",
            "tokenizer": {
                "type": "word",
                "word_splitter": {
                    "type": "just_spaces"
                }
            }
        },
        "extra_candidate_generators": {
            "wordnet": {
                "type": "wordnet_mention_generator",
                "entity_file": "s3://allennlp/knowbert/wordnet/entities.jsonl"
            }
        },
        "should_remap_span_indices": True,
        "token_indexers": {
            "tokens": {
                "type": "bert-pretrained",
                "do_lowercase": True,
                "max_pieces": 512,
                "pretrained_model": "bert-base-uncased",
                "use_starting_offsets": True,
            }
        }
    })

reader_params_w = Params({
    "type": "aida_wiki_linking",
    "entity_disambiguation_only": False,
    "token_indexers": {
        "tokens": {
            "type": "bert-pretrained",
            "pretrained_model": "bert-base-uncased",
            "do_lowercase": True,
            "use_starting_offsets": True,
            "max_pieces": 512,
        },
    },
    "entity_indexer": {
        "type": "characters_tokenizer",
        "tokenizer": {
            "type": "word",
            "word_splitter": {"type": "just_spaces"},
        },
        "namespace": "entity",
    },
    "should_remap_span_indices": True,
})

cg_params_ww = Params({
                "type": "bert_tokenizer_and_candidate_generator",
                "bert_model_type": "bert-base-uncased",
                "do_lower_case": True,
                "entity_candidate_generators": {
                    "wordnet": {
                        "type": "wordnet_mention_generator",
                        "entity_file": "s3://allennlp/knowbert/wordnet/entities.jsonl"
                    }
                },
                "entity_indexers": {
                    "wordnet": {
                        "type": "characters_tokenizer",
                        "namespace": "entity_wordnet",
                        "tokenizer": {
                            "type": "word",
                            "word_splitter": {
                                "type": "just_spaces"
                            }
                        }
                    }
                }
            })

def decode(linking_scores, candidate_spans, candidate_entities):

    # Placeholder for the best entities
    best_entities = []

    # Iterate over each span in candidate_spans
    for i, span in enumerate(candidate_spans):
        # Extracting the linking scores for the current span
        entity_scores = linking_scores[i]
        entities = candidate_entities[i]

        # Finding the index of the highest score for both wiki and wordnet
        best_index = entity_scores.argmax().item()

        # Fetching the best entity for the current span using the index
        best_entity = entities[best_index]

        # Fetching the best scores for the current span
        best_score = entity_scores[best_index].item()

        # Appending the best entities to the placeholder
        best_entities.append({"span": span, "entity": best_entity, "score": best_score})

    return best_entities

def map_entity_ids_to_names(best_entities_and_scores, entity_names, entity_ids_tensor):
    # Convert tensor to list for easier indexing
    entity_ids = entity_ids_tensor[0].tolist()

    # Placeholder for the mapped entities
    mapped_entities = []

    # Iterate over each span's best entities and scores
    for i, entity_data in enumerate(best_entities_and_scores):
        # Extracting the best entity names using the entity IDs
        entity_id = entity_data['entity']
        span = entity_data["span"]
        score = entity_data["score"]

        # Map the entity IDs to their respective names
        entity_name = entity_names[i][entity_ids[i].index(entity_id)] if entity_id in entity_ids[i] else None

        # Appending the mapped entities and their scores to the placeholder
        mapped_entities.append({
            'span': span.cpu().numpy().tolist(),
            'entity': entity_name,
            'entity_id': entity_id.cpu().item(),
            'score': score,
        })

    return mapped_entities

def filter_by_score(mapped_entities, threshold):
    return [entry for entry in mapped_entities if entry['score'] >= threshold]


def get_word_span(token_span, token_spans, word_spans):
    """
    Given a token span, return the corresponding word span.

    :param token_span: The token span for which we want the corresponding word span.
    :param token_spans: Tensor containing token spans.
    :param word_spans: List of word spans.
    :return: Corresponding word span.
    """
    # Convert token_span to a tensor and ensure it's on the CPU
    token_span_tensor = torch.tensor(token_span).cpu()
    
    # Ensure token_spans is also on the CPU
    token_spans_cpu = token_spans.cpu()
    
    # Find the index of the token_span in token_spans
    matches = (token_spans_cpu[0] == token_span_tensor).all(dim=1)
    index = torch.nonzero(matches, as_tuple=True)[0]
    
    # If the token span is not found, return None
    if len(index) == 0:
        return None
    
    # Return the corresponding word span
    return word_spans[index[0].item()]

def map_entity_id_to_name(target_id, entity_ids, entity_names):
    """
    Retrieve the name corresponding to the given ID.

    :param entity_names: 2D list containing names of entities.
    :param entity_ids: 2D list containing IDs of entities.
    :param target_id: The ID for which we want the corresponding name.
    :return: Corresponding name or None if not found.
    """
    ids_list = entity_ids[0].tolist()

    for i, row in enumerate(ids_list):
        for j, entity_id in enumerate(row):
            if entity_id == target_id:
                return entity_names[i][j]
    return None

def word_to_char_span(word_span, words):
    # Compute the start position of each word
    word_starts = [0]
    for word in words[:-1]:
        word_starts.append(word_starts[-1] + len(word) + 1)  # +1 for the space
    
    start = word_span[0]
    end = word_span[1]
    char_start = word_starts[start]
    char_end = word_starts[end] + len(words[end])  # add -1 if end index is inclusive

    return (char_start, char_end)

def convert_decode_output(decode_output, entity_ids, entity_names, token_spans, word_spans, words):
    '''
    input: [(0, (1, 2), 320843), (0, (7, 8), 426103), ... ]
    '''
    converted_output = []
    for _, token_span, entity_id in decode_output:
        entity_name = map_entity_id_to_name(entity_id, entity_ids, entity_names)
        word_span = get_word_span(token_span, token_spans, word_spans)
        character_span = word_to_char_span(word_span, words)
        mention = (character_span[0], character_span[1], entity_name)
        converted_output.append(mention)
    return converted_output


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--model_archive', type=str)
parser.add_argument('--wiki_and_wordnet', action='store_true')

args = parser.parse_args()

model_archive_file = args.model_archive
is_wordnet_and_wiki = args.wiki_and_wordnet

print("Done arg parse")

archive = load_archive(model_archive_file)
params = archive.config
vocab = Vocabulary.from_params(params.pop('vocabulary'))

model = archive.model
model.cuda()
model.eval()

wiki_el = getattr(model, "wiki_soldered_kg").entity_linker

if is_wordnet_and_wiki:
    wordnet_el = getattr(model, "wordnet_soldered_kg").entity_linker

print("Loaded model.")

if is_wordnet_and_wiki:
    reader_params = reader_params_ww
else:
    reader_params = reader_params_w

if is_wordnet_and_wiki:
    candidate_generator = TokenizerAndCandidateGenerator.from_params(cg_params_ww)

reader = DatasetReader.from_params(Params(reader_params))

iterator = DataIterator.from_params(Params({"type": "basic", "batch_size": 1}))
iterator.index_with(vocab)

print("Got candidate generator and reader.")

#elb = EntityLinkingBase(vocab)

THE_INPUT_SENTENCE = "CRICKET - LEICESTERSHIRE TAKE OVER AT TOP AFTER INNINGS VICTORY. LONDON 1996-08-30 West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset by an innings and 39 runs in two days to take over at the head of the county championship. Their stay on top, though, may be short-lived as title rivals Essex, Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire. After bowling Somerset out for 83 on the opening morning at Grace Road, Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83. Trailing by 213, Somerset got a solid start to their second innings before Simmons stepped in to bundle them out for 174. Essex, however, look certain to regain their top spot after Nasser Hussain and Peter Such gave them a firm grip on their match against Yorkshire at Headingley. Hussain, considered surplus to England's one-day requirements, struck 158, his first championship century of the season, as Essex reached 372 and took a first innings lead of 82. By the close Yorkshire had turned that into a 37-run advantage but off-spinner Such had scuttled their hopes, taking four for 24 in 48 balls and leaving them hanging on 119 for five and praying for rain. At the Oval, Surrey captain Chris Lewis, another man dumped by England, continued to silence his critics as he followed his four for 45 on Thursday with 80 not out on Friday in the match against Warwickshire. He was well backed by England hopeful Mark Butcher who made 70 as Surrey closed on 429 for seven, a lead of 234. Derbyshire kept up the hunt for their first championship title since 1936 by reducing Worcestershire to 133 for five in their second innings, still 100 runs away from avoiding an innings defeat. Australian Tom Moody took six for 82 but Chris Adams, 123, and Tim O'Gorman, 109, took Derbyshire to 471 and a first innings lead of 233. After the frustration of seeing the opening day of their match badly affected by the weather, Kent stepped up a gear to dismiss Nottinghamshire for 214. They were held up by a gritty 84 from Paul Johnson but ex-England fast bowler Martin McCague took four for 55. By stumps Kent had reached 108 for three."
processed = reader.mention_generator.get_mentions_raw_text(THE_INPUT_SENTENCE, whitespace_tokenize=True)
# get_mentions_raw_text returns a very slightly different dict from get_mentions_with_gold
processed["candidate_entity_prior"] = processed["candidate_entity_priors"]
del processed["candidate_entity_priors"]
instance = reader.text_to_instance(doc_id="", **processed)
instances = [instance]

print("Processed the sentence.")

for batch_no, batch in enumerate(iterator(instances, shuffle=False, num_epochs=1)):
    b = move_to_device(batch, 0)

    b['candidates'] = {'wiki': {
            'candidate_entities': b.pop('candidate_entities'),
            'candidate_entity_priors': b.pop('candidate_entity_prior'),
            'candidate_segment_ids': b.pop('candidate_segment_ids'),
            'candidate_spans': b.pop('candidate_spans')}}

    if is_wordnet_and_wiki:
        extra_candidates = b.pop('extra_candidates')
        seq_len = b['tokens']['tokens'].shape[1]
        bbb = []
        for e in extra_candidates:
            for k in e.keys():
                e[k]['candidate_segment_ids'] = [0] * len(e[k]['candidate_spans'])
            ee = {'tokens': ['[CLS]'] * seq_len, 'segment_ids': [0] * seq_len,
                    'candidates': e}
            ee_fields = candidate_generator.convert_tokens_candidates_to_fields(ee)
            bbb.append(Instance(ee_fields))
        eb = Batch(bbb)
        eb.index_instances(vocab)
        padding_lengths = eb.get_padding_lengths()
        tensor_dict = eb.as_tensor_dict(padding_lengths)
        b['candidates'].update(tensor_dict['candidates'])
        bb = move_to_device(b, 0)
    else:
        bb = b

    raw_output = model(**bb)

    wiki_decoded_output = wiki_el._decode(raw_output["wiki"]["linking_scores"], bb["candidates"]["wiki"]["candidate_spans"], bb["candidates"]["wiki"]["candidate_entities"]["ids"])

    if is_wordnet_and_wiki:
        wordnet_decoded_output = wordnet_el._decode(raw_output["wordnet"]["linking_scores"], bb["candidates"]["wordnet"]["candidate_spans"], bb["candidates"]["wordnet"]["candidate_entities"]["ids"])
    
    converted_output = convert_decode_output(wiki_decoded_output, bb["candidates"]["wiki"]["candidate_entities"]["ids"], processed["candidate_entities"], bb["candidates"]["wiki"]["candidate_spans"], processed["candidate_spans"], processed["tokenized_text"])

    with open('converted_output.pkl', 'wb') as file:
        pickle.dump({"processed": processed, "batch": batch, "bb": bb, "raw_output": raw_output, "wiki_decoded_output": wiki_decoded_output, "wordnet_decoded_output": wordnet_decoded_output, "converted_output": converted_output}, file)
    
    print(converted_output)
    
    '''
    print("RAW")
    print(raw_output)

    with open('pbbr.pkl', 'wb') as file:
        pickle.dump({"processed": processed, "batch": batch, "bb": bb, "raw_output": raw_output}, file)

    wiki_output = decode(raw_output["wiki"]["linking_scores"][0], bb["candidates"]["wiki"]["candidate_spans"][0], bb["candidates"]["wiki"]["candidate_entities"]["ids"][0])
    wiki_output_names = map_entity_ids_to_names(wiki_output, processed["candidate_entities"], bb["candidates"]["wiki"]["candidate_entities"]["ids"])
    wiki_final_output = filter_by_score(wiki_output_names, 0)
    
    if is_wordnet_and_wiki:
        wordnet_output = decode(raw_output["wordnet"]["linking_scores"][0], bb["candidates"]["wordnet"]["candidate_spans"][0], bb["candidates"]["wordnet"]["candidate_entities"]["ids"][0])
        wordnet_output_names = map_entity_ids_to_names(wordnet_output, batch["extra_candidates"][0]["wordnet"]["candidate_entities"], bb["candidates"]["wordnet"]["candidate_entities"]["ids"])
        wordnet_final_output = filter_by_score(wordnet_output_names, 5)
    
    print(THE_INPUT_SENTENCE)
    print(wiki_final_output)
    print(wordnet_final_output)
    '''
