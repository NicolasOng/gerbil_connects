from gerbil_connect.server_template import eeebert_model

import pickle

with open("gold_documents_new.pkl", 'rb') as file:
    new_gold_documents = pickle.load(file)

for doc in new_gold_documents:
    output = eeebert_model(doc["raw_text"])
    print(output)
    break