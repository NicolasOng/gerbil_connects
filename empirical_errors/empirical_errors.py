import argparse
import json

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

def spans_overlap(span1, span2):
    """
    Check if two spans overlap.
    Each span is {'start': 5, 'end': 10, 'tag': '...'}
    """
    start1, end1 = span1['start'], span1['end']
    start2, end2 = span2['start'], span2['end']
    return max(start1, start2) < min(end1, end2)

def overlapping_spans_exist(span, span_list):
    '''
    Checks if there is an overlapping span in the list.
    If so, returns the first one by start index.
    Each span is {'start': 5, 'end': 10, 'tag': '...'}
    '''
    overlapping_spans = []
    for lspan in span_list:
        if spans_overlap(span, lspan):
            overlapping_spans.append(lspan)
    if len(overlapping_spans) == 0: return False, None
    overlapping_spans.sort(key=lambda span: span['start'])
    return True, overlapping_spans[0]

def first_close_span(span, span_list):
    '''
    gets the index of first span in the list with an equal or above starting value.
    '''
    span_list.sort(key=lambda span: span['start'])
    index = -1
    for i, lspan in enumerate(span_list):
        if lspan['start'] >= span['start']:
            index = i
            break
    return index

def get_entity(link):
    return link['tag'].rsplit('/', 1)[-1]

def span_to_text(span, text):
    return text[span['start']-10:span['start']] + "*" + text[span['start']:span['end']] + "*" + text[span['end']:span['end']+10]

def span_to_text_raw(span):
    return f"{span['start']}, {span['end']}, {get_entity(span)}"

def print_span(span, text):
    print(span_to_text_raw(span))
    print(span_to_text(span, text))

parser = argparse.ArgumentParser()
parser.add_argument('--gold', type=str)
parser.add_argument('--pred', type=str)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

gold_file = args.gold
pred_file = args.pred
verbose = args.verbose

gold_annotations = read_json_file(gold_file)["annotations"]
pred_annotations = read_json_file(pred_file)["annotations"]

print(f"Num documents in gold/pred annotations: {len(gold_annotations)}/{len(pred_annotations)}")

texts_in_gold = set(item["text"] for item in gold_annotations)
texts_in_pred = set(item["text"] for item in pred_annotations)

print(f"Num unique documents in gold/pred annotations: {len(texts_in_gold)}/{len(texts_in_pred)}")

pred_annotations = [item for item in pred_annotations if item["text"] in texts_in_gold]
gold_annotations = [item for item in gold_annotations if item["text"] in texts_in_pred]

print(f"Num documents in gold/pred annotations after filter: {len(gold_annotations)}/{len(pred_annotations)}")

gold_annotations = sorted(gold_annotations, key=lambda x: x['text'])
pred_annotations = sorted(pred_annotations, key=lambda x: x['text'])

print("First pred document: " + pred_annotations[0]["text"][:50] + "...")
print("First gold document: " + gold_annotations[0]["text"][:50] + "...")

assert len(gold_annotations) == len(pred_annotations)

num_gold_mentions = 0
mentions_predicted = 0
overgeneration = 0
undergeneration = 0
wrong_entity = 0
wrong_mention = 0

for i in range(len(gold_annotations)):
    gold_annotation = gold_annotations[i]
    pred_annotation = pred_annotations[i]

    assert_msg = f"Document {i}: {gold_annotation['text'][:50]}/{pred_annotation['text'][:50]}"
    assert gold_annotation['text'] == pred_annotation['text'], assert_msg

    text = gold_annotation['text']
    gold_spans = gold_annotation['value']
    pred_spans = pred_annotation['value']

    gold_spans.sort(key=lambda span: span['start'])
    pred_spans.sort(key=lambda span: span['start'])

    if verbose:
        print(f"Document {i}: {text[:100]}...")
        print(f"Num gold/pred spans: {len(gold_spans)}/{len(pred_spans)}\n")
    
    num_gold_mentions += len(gold_spans)
    mentions_predicted += len(pred_spans)

    # overgeneration
    for pred in pred_spans:
        #if an overlapping gold span doesn't exist, found an overgeneration.
        overlaps, _ = overlapping_spans_exist(pred, gold_spans)
        if not overlaps:
            overgeneration += 1
            if verbose:
                close_idx = first_close_span(pred, gold_spans)
                c1_span = gold_spans[close_idx - 1]
                c2_span = gold_spans[close_idx]
                print("Found overgeneration: ")
                print("Predicted Span: ")
                print_span(pred, text)
                print("Two closest gold spans: ")
                print_span(c1_span, text)
                print_span(c2_span, text)
                print("")


    #undergeneration
    for gold in gold_spans:
        #if an overlapping pred span doesn't exist, found an undergeneration.
        overlaps, _ = overlapping_spans_exist(gold, pred_spans)
        if not overlaps:
            undergeneration += 1
            if verbose:
                close_idx = first_close_span(gold, pred_spans)
                c1_span = pred_spans[close_idx - 1]
                c2_span = pred_spans[close_idx]
                print("Found undergeneration: ")
                print("Gold Span: ")
                print_span(gold, text)
                print("Two closest predicted spans: ")
                print_span(c1_span, text)
                print_span(c2_span, text)
                print("")

    # incorrect entity link
    for pred in pred_spans:
        # if the first overlapping gold span (if there is one) has a different entity, found an incorrect entity link.
        overlaps, overlapping_span = overlapping_spans_exist(pred, gold_spans)
        if overlaps:
            if get_entity(pred) != get_entity(overlapping_span):
                wrong_entity += 1
                if verbose:
                    print("Found Incorrect Entity Link: ")
                    print("Predicted Span: ")
                    print_span(pred, text)
                    print("Gold Span: ")
                    print_span(overlapping_span, text)
                    print("")

    # incorrect mention prediction:
    for pred in pred_spans:
        # if the first overlapping gold span (if there is one) has a different span, found an incorrect mention prediction.
        overlaps, overlapping_span = overlapping_spans_exist(pred, gold_spans)
        if overlaps:
            if not (pred['start'] == overlapping_span['start'] and pred['end'] == overlapping_span['end']):
                wrong_mention += 1
                if verbose:
                    print("Found Incorrect Mention Span: ")
                    print("Predicted Span: ")
                    print_span(pred, text)
                    print("Gold Span: ")
                    print_span(overlapping_span, text)
                    print("")

print("Results: ")
print(f"Number of Gold Mentions (after documents the model errored on were removed): {num_gold_mentions}\nMentions Predicted: {mentions_predicted}\nOvergeneration: {overgeneration}\nUndergeneration: {undergeneration}\nIncorrect Entity Link: {wrong_entity}\nIncorrect Mention Span: {wrong_mention}")
print(f"{num_gold_mentions}\t{mentions_predicted}\t{overgeneration}\t{undergeneration}\t{wrong_entity}\t{wrong_mention}")
