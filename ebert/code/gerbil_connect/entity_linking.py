import pickle

from gerbil_connect.server_template import ebert_model

with open('aida_gold_documents.pkl', 'rb') as file:
    gold_documents = pickle.load(file)

sentence = gold_documents[0]["raw_text"]

output = ebert_model(sentence)

print(output)
