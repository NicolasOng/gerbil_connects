import json
import pickle

def replace_asterisks(jsonl_file, out_file, docs_file):

    with open(docs_file, 'rb') as file:
        new_gold_documents = pickle.load(file)

    # Output list to store updated JSON objects
    output_data = []

    # Read the JSONL file line by line
    with open(jsonl_file, 'r') as f:
        for line_i, line in enumerate(f):
            print(line_i)
            # get the text and length on that line
            data = json.loads(line)
            text = data["text"]
            text_len = len(text)

            # get the unobfuscated candidates - matching length.
            candidates = []
            for doc in new_gold_documents:
                doc_text = doc["doc_spaces"]
                doc_text_len = len(doc_text)
                if (text_len == doc_text_len):
                    candidates.append(doc_text)
            print(len(candidates))
            # to deal with multiple candidates, go character by character.
            # eg: there are 3 docs with len 1214
            for i, char in enumerate(text):
                if char == "*": continue
                for j in range(len(candidates)-1, -1, -1):  # reverse iteration
                    candidate = candidates[j]
                    cand_char = candidate[i]
                    if (char != cand_char):
                        del candidates[j]
                    if len(candidates) <= 1:
                        break
            
            print("OBFUSCATED TEXT: ", data["text"])
            if len(candidates) < 1:
                print("ERROR: NO CANDIDATES.")
                print(" ")
                raise Exception("No Candidates!!!")
            else:
                print("UNOBFUSCATED TEXT: ", candidates[0])
                print(" ")
            
            # Update the text field with the modified string
            data["text"] = candidates[0]
            output_data.append(data)
            
    # Write the updated data to a new JSONL file
    with open(out_file, 'w') as f:
        for item in output_data:
            f.write(json.dumps(item) + '\n')


#replace_asterisks('data/benchmarks/aida-conll-dev.benchmark.jsonl', 'data/benchmarks/aida-conll-dev.benchmark.unobfuscated.jsonl', "gold_documents_new_02.pkl")
replace_asterisks('data/benchmarks/aida-conll-test.benchmark.jsonl', 'data/benchmarks/aida-conll-test.benchmark.unobfuscated.jsonl', "gold_documents_new_02.pkl")
