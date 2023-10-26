from gerbil_connect.server_template import ea_el_model

import pickle
with open("gold_documents_new.pkl", 'rb') as file:
    new_gold_documents = pickle.load(file)


for i, item in enumerate(new_gold_documents):
    #if (i != 145): continue
    print("--------", i)
    raw_text = item["raw_text"]

    output = ea_el_model(raw_text)

    print(output)
    break
