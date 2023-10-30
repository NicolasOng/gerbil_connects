from gerbil_connect.server_template import ebert_model

import pickle

def print_results(raw_text, output):
    for start, end, entity in output:
        print(start, end, entity)
        print(raw_text[start:end])
        print(raw_text[start-10:start] + "*" + raw_text[start:end] + "*" + raw_text[end:end+10])
        print("")

with open("gold_documents_new.pkl", 'rb') as file:
    new_gold_documents = pickle.load(file)

# Initialize an empty dictionary to store the outputs
outputs_dict = {}

for i, doc in enumerate(new_gold_documents):
    print("------ " + str(i) + " ------")
    raw_text = doc["raw_text"]
    output = ebert_model(raw_text)
    outputs_dict[raw_text] = output
    print(raw_text)
    print(output)
    print_results(raw_text, output)

if True:
    with open('outputs_dict.pkl', 'wb') as f:
        pickle.dump(outputs_dict, f)
    print('Outputs saved to outputs_dict.pkl')
