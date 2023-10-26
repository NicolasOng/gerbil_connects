from gerbil_connect.server_template import knowbert_model

def print_results(raw_text, output):
    for start, end, entity in output:
        print(start, end, entity)
        print(raw_text[start:end])
        print(raw_text[start-10:start] + "*" + raw_text[start:end] + "*" + raw_text[end:end+10])
        print("")

import pickle
with open('gold_documents_new_02.pkl', 'rb') as file:
    gold_documents = pickle.load(file)

for i, document in enumerate(gold_documents):
    if i != 0:
        continue
    print(i)
    raw_text = document["raw_text"]
    
    output = knowbert_model(raw_text)

    print_results(raw_text, output)
