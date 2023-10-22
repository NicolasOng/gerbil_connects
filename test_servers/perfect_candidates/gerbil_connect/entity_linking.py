import pickle

from gerbil_connect.server_template import perfect_entity_linking_model

with open('gold_documents_new_02.pkl', 'rb') as file:
    gold_documents = pickle.load(file)

sentence = gold_documents[0]["raw_text"]
outputs = perfect_entity_linking_model(sentence)

print(outputs)
