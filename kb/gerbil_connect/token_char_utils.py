import re

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

def remove_whitespaces(s):
    '''
    removes whitespaces - spaces, tabs, newlines.
    '''
    return re.sub(r'\s+', '', s)

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
