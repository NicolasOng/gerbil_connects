import re
import pickle

with open('aida_gold_documents.pkl', 'rb') as file:
    gold_documents = pickle.load(file)

def remove_whitespaces(s):
    '''
    removes whitespaces - spaces, tabs, newlines.
    '''
    return re.sub(r'\s+', '', s)

def split_into_sentences(text):
    '''
    Simple function to split text into its sentences.
    Made for models that can't take in the whole text.
    Doesn't handle complex cases like Dr. or initials.
    '''
    # Split the text by any of the sentence-ending punctuation marks
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    # Remove any empty strings from the list
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def split_by_whitespace(text, max_len=500):
    '''
    Simple function for models that require text to be given
    in a ["Hello", "my", "name", "is", ...] format.
    Just splits by white spaces (a whitespace tokenizer).
    '''
    tokens = text.split()
    tokens = [word for word in tokens if word.strip()]

    return tokens

def split_by_max_len(tokens, max_len=500):
    '''
    splits a list of tokens like ["Hello", "my", "name", "is", ...]
    into multiple lists, if it's too long.
    '''
    # If the tokens list is longer than max_len, split it into smaller chunks
    if len(tokens) > max_len:
        num_chunks = -(-len(tokens) // max_len)  # This is a ceiling division trick
        chunked_tokens = [tokens[i * max_len: (i + 1) * max_len] for i in range(num_chunks)]
        return chunked_tokens
    return [tokens]

def character_to_character_index(s1, s2, idx):
    '''
    input: two strings with the same non-whitespace characters, and the character index of a non-whitespace character in the first string.
    output: the index of that same character in the second string.
    Use:
    When two systems have different tokenizers, so one does
    "There was a horse , cow , and seagull ."
    and the other does
    "There was a horse, cow, and seagull."
    and you have the index for one, but need the other.
    '''
    # Check if the index is valid for the first string
    if idx < 0 or idx >= len(s1):
        raise ValueError("Invalid index for the first string.")
    
    # Find the character in the first string using the given index
    char_in_s1 = s1[idx]
    
    # If the character is a whitespace, return an error
    if char_in_s1.isspace():
        raise ValueError("The character at the given index is a whitespace.")
    
    # Count the occurrence of the character in s1 up to the given index
    occurrence_in_s1 = s1[:idx+1].count(char_in_s1)
    
    # Iterate over the second string to find the index of that character
    occurrence_in_s2 = 0
    for i, char in enumerate(s2):
        if char == char_in_s1:
            occurrence_in_s2 += 1
            if occurrence_in_s2 == occurrence_in_s1:
                return i
    raise ValueError(f"Character '{char_in_s1}' not found in the second string for the {occurrence_in_s1}th time.")

def token_to_character_index(tokens, characters, start_idx, end_idx):
    '''
    given:
    ["Hello", ",", "how", "are", "you", "?"], "Hello,     how are you?", 2, 5
    returns:
    11, 21
    Which refer to the asterisk'd characters in:
    "Hello,     *h*ow are yo*u*?"
    Note the large amount of spaces.
    '''
    char_start_nws_idx, char_end_nws_idx = token_to_no_whitespace_character_index(tokens, start_idx, end_idx)
    nws_tokens = ''.join(tokens)
    char_start_idx = character_to_character_index(nws_tokens, characters, char_start_nws_idx)
    char_end_idx = character_to_character_index(nws_tokens, characters, char_end_nws_idx)
    return char_start_idx, char_end_idx

def token_to_no_whitespace_character_index(tokens, start_token_index, end_token_index):
    '''
    given:
    ["Hello", ",", "how", "are", "you", "?"], 2, 5
    returns:
    6, 14
    Which refer to the asterisk'd characters in the following string:
    "Hello,*h*owareyo*u*?"
    '''
    # Initial index
    idx = 0
    
    # Iterate through the tokens up to the end_token_index
    for i, token in enumerate(tokens):
        # If we've reached the start token, record the start index
        if i == start_token_index:
            start_idx = idx
        
        # Move the index by the length of the current token
        idx += len(token)
        
        # If we've reached the end token, record the end index and break
        if i == end_token_index:
            end_idx = idx - 1
            break
    
    return start_idx, end_idx

def aida_get_gold_document(raw_text):
    '''
    given the raw text from GERBIL, this gets the "gold document" from a file.
    It includes:
    - words: the tokenized sentence. eg: ['CRICKET', '-', 'ENGLISH', ...]
    - doc_no_whitespace: ''.join(words)
    - doc_spaces: ' '.join(words)
    - gold_spans: refers to the tokens. eg: [[2, 2], [13, 14], ...]
    - gold_entities: same length as the span list. eg: ['London', 'Phil_Simmons', ...]
    '''
    target_string = remove_whitespaces(raw_text)
    for doc in gold_documents:
        if doc["doc_no_whitespace"] == target_string:
            matching_document = doc
            break
    if matching_document is None:
        print("No matching document found for: \n" + target_string)
    return matching_document

def aida_tokenize(raw_text):
    '''
    "tokenizes" the input exactly as AIDA would.
    '''
    gold_doc = aida_get_gold_document(raw_text)
    return gold_doc["words"]