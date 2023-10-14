import json

# Step 1: Load the TSV file into a dictionary
qid_to_name = {}
with open('data/elevant/qid_to_wikipedia_url.tsv', 'r') as tsv_file:
    for line in tsv_file:
        qid, url = line.strip().split('\t')
        name = url.split('/')[-1]  # Extract Wikipedia name from the URL
        qid_to_name[qid] = name

def do_jsonl(in_f, out_f):
    # Step 2: Parse the JSONL file line by line
    with open(in_f, 'r') as jsonl_file, open(out_f, 'w') as output_file:
        for line in jsonl_file:
            data = json.loads(line)

            # Step 3: Iterate over the entity_mentions list
            for entity in data['entity_mentions']:
                # Step 4: Add the wiki id to the entity
                entity['wiki_id'] = qid_to_name.get(entity['id'], "")

            # Step 5: Write the modified line back
            output_file.write(json.dumps(data) + '\n')

#do_jsonl("dev-out.qids.jsonl", "dev-out.wiki_ids.jsonl")
#do_jsonl("test-out.qids.jsonl", "test-out.wiki_ids.jsonl")
