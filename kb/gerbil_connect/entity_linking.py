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

from gerbil_connect.helper_functions import aida_get_gold_document, sentence_tokenize, punkt_tokenize, token_to_character_index, character_to_character_index

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

def split_at_periods(tokens, spans, entities):
    '''
    inputs:
        tokens = ["John", "Adams", "is", "at", "the", "game", ".", "He"]
        spans = [[0, 1], [7, 7]]
        entities = ["John_Adams", "John_Adams"]
    outputs:
        [['John', 'Adams', 'is', 'at', 'the', 'game', '.'], ['He']]
        [[[0, 1]], [[7, 7]]]
        [['John_Adams'], ['John_Adams']]
    '''
    split_tokens = []
    split_spans = []
    split_entities = []

    current_tokens = []
    current_spans = []
    current_entities = []

    token_start_idx = 0  # relative starting index of the current sentence

    for i, token in enumerate(tokens):
        current_tokens.append(token)

        # Check if current span is within the current sentence
        if spans and spans[0][0] - token_start_idx < len(current_tokens):
            start, end = spans.pop(0)
            adjusted_span = [start - token_start_idx, end - token_start_idx]
            current_spans.append(adjusted_span)
            current_entities.append(entities.pop(0))

        # Check if token is a period
        if token == ".":
            if len(current_tokens) < 2 and split_tokens:
                # Merge the current_tokens with the last entry in split_tokens
                split_tokens[-1].extend(current_tokens)
                
                # Also merge spans and entities if any
                if current_spans:
                    split_spans[-1].extend(current_spans)
                    split_entities[-1].extend(current_entities)
            else:
                split_tokens.append(current_tokens)
                split_spans.append(current_spans)
                split_entities.append(current_entities)
            current_tokens = []
            current_spans = []
            current_entities = []
            token_start_idx = i + 1

    # Add any remaining tokens, spans, and entities
    if current_tokens:
        # Adjusting remaining spans
        for j, (start, end) in enumerate(current_spans):
            current_spans[j] = [start - token_start_idx, end - token_start_idx]
            
        if len(current_tokens) < 2 and split_tokens:
            # Merge the current_tokens with the last entry in split_tokens
            split_tokens[-1].extend(current_tokens)
            
            # Also merge spans and entities if any
            if current_spans:
                split_spans[-1].extend(current_spans)
                split_entities[-1].extend(current_entities)
        else:
            split_tokens.append(current_tokens)
            split_spans.append(current_spans)
            split_entities.append(current_entities)

    return split_tokens, split_spans, split_entities

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

def convert_final_predictions(predictions, pred_sentence, new_sentence):
    '''
    input: [(12, 35, "Entity_Name"), ...]
    Where the span indices refer to characters in the pred_sentence,
    but we want them to refer to the characters in the new_sentence.
    '''
    new_preds = []
    for pred in predictions:
        start_pred = pred[0]
        end_pred = pred[1]
        pred_name = pred[2]
        new_start = character_to_character_index(pred_sentence, new_sentence, start_pred)
        new_end = character_to_character_index(pred_sentence, new_sentence, end_pred - 1) + 1
        new_preds.append((new_start, new_end, pred_name))
    return new_preds

with open('gold_documents_new_02.pkl', 'rb') as file:
    gold_documents = pickle.load(file)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--model_archive', type=str)
parser.add_argument('--wiki_and_wordnet', action='store_true')
parser.add_argument('--no-candidate-sets', action='store_true')

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

print("Loaded model.")

if is_wordnet_and_wiki:
    reader_params = reader_params_ww
else:
    reader_params = reader_params_w

if is_wordnet_and_wiki:
    candidate_generator = TokenizerAndCandidateGenerator.from_params(cg_params_ww)

reader = DatasetReader.from_params(Params(reader_params))

if (args.no_candidate_sets):
        print("no candidate sets.")
        reader.set_no_candidate_sets()

iterator = DataIterator.from_params(Params({"type": "basic", "batch_size": 1}))
iterator.index_with(vocab)

print("Got candidate generator and reader.")

for i, document in enumerate(gold_documents):
    if i != 0:
        continue
    print(i)
    raw_text = document["raw_text"]
    gold = False

    instances = []
    processeds = []

    if gold:
        # get the gold aida tokenization, spans, and entities
        aida_document = aida_get_gold_document(raw_text)
        # break it down by sentences - break at the periods
        # span indices need to be updated.
        tokenized_sentences, sentence_gold_spans, sentence_gold_entities = split_at_periods(aida_document["words"], aida_document["gold_spans"], aida_document["gold_entities"])
        for i, tokenized_sentence in enumerate(tokenized_sentences):
            gold_spans = sentence_gold_spans[i]
            gold_entities = sentence_gold_entities[i]
            # creates candidate spans + entities, but adds the gold spans to the candidate list.
            processed = reader.mention_generator.get_mentions_with_gold(' '.join(tokenized_sentence), gold_spans, gold_entities, whitespace_tokenize=True, keep_gold_only=False)
            # converts to Allennlp Instance, adds extra span/entity candidates for wordnet
            instance = reader.text_to_instance(doc_id="", **processed)
            processeds.append(processed)
            instances.append(instance)
    else:
        # break the given text into sentences
        sentences = sentence_tokenize(raw_text)
        # tokenize the sentences
        tokenized_sentences = [punkt_tokenize(sentence) for sentence in sentences]
        for tokenized_sentence in tokenized_sentences:
            # creates candidate token spans + entities.
            processed = reader.mention_generator.get_mentions_raw_text(' '.join(tokenized_sentence), whitespace_tokenize=True)
            # get_mentions_raw_text returns a very slightly different dict from get_mentions_with_gold
            processed["candidate_entity_prior"] = processed["candidate_entity_priors"]
            del processed["candidate_entity_priors"]
            # converts to Allennlp Instance, adds extra span/entity candidates for wordnet
            instance = reader.text_to_instance(doc_id="", **processed)
            processeds.append(processed)
            instances.append(instance)

    total_final_output = []
    text_predicted_for = ""
    for batch_no, batch in enumerate(iterator(instances, shuffle=False, num_epochs=1)):
        # b is the Allennlp Instance after the iterator turns it into a model-friendly input.
        # Converts the word-level tokens to BERT tokens, adjusts the span indices, converts entity names to ids, etc.
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

        # bb is b after some processing (see above).

        # output has linking scores for every entity.
        raw_output = model(**bb)

        # create some vars for convinience:
        linking_scores = raw_output["wiki"]["linking_scores"]

        entity_ids = bb["candidates"]["wiki"]["candidate_entities"]["ids"]
        entity_names = processeds[batch_no]["candidate_entities"]

        bert_token_spans = bb["candidates"]["wiki"]["candidate_spans"]
        word_token_spans = processeds[batch_no]["candidate_spans"]

        tokenized_sentence = processeds[batch_no]["tokenized_text"]
        sentence = " ".join(tokenized_sentence)

        # input: list of candidate spans, list of candidate entities for each span, "linking scores" for each entity
        # decode returns a list of [batch, (start, end), entity id].
        # the start/end refer to the BERT tokens.
        decoded_output = wiki_el._decode(linking_scores, bert_token_spans, entity_ids)

        final_output = []
        # for each predicted span/entity,
        for _, token_span, entity_id in decoded_output:
            # convert the entity id to the entity name
            entity_name = map_entity_id_to_name(entity_id, entity_ids, entity_names)
            # convert the BERT token span to the word token span
            word_span = get_word_span(token_span, bert_token_spans, word_token_spans)
            # convert the word token span to a character span
            character_span = token_to_character_index(tokenized_sentence, sentence, word_span[0], word_span[1])
            # convert to sentence-level character span to a document-level character span
            start_i = character_span[0] + len(text_predicted_for)
            end_i = character_span[1] + len(text_predicted_for) + 1
            # append the mention.
            mention = (start_i, end_i, entity_name)
            final_output.append(mention)
        
        # update the document predicted for
        text_predicted_for += sentence

        total_final_output += final_output

    # the text that was predicted for might be different from the text given.
    total_final_output = convert_final_predictions(total_final_output, text_predicted_for, raw_text)

    print("number of mentions detected: ", len(total_final_output))

    for start, end, entity in total_final_output:
        print(start, end, entity)
        print(raw_text[start:end])
        print(raw_text[start-10:start] + "*" + raw_text[start:end] + "*" + raw_text[end:end+10])
        print("")

'''
with open('converted_output.pkl', 'wb') as file:
    pickle.dump({"processed": processed, "batch": batch, "bb": bb, "raw_output": raw_output, "wiki_decoded_output": wiki_decoded_output, "wordnet_decoded_output": wordnet_decoded_output, "converted_output": converted_output}, file)

print(converted_output)
'''