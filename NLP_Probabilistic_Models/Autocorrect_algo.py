# Part 1 data preprocessing
import re
from collections import Counter
import numpy as np
import pandas as pd


# Process functions
def process_data(file_name):
    """
    Input:
        A file_name which is found in your current directory. You just have to read it in.
    Output:
        words: a list containing all the words in the corpus (text file you read) in lower case.
    """
    words = []  # return this variable correctly

    df = pd.read_csv(file_name, sep="\t", header=None)
    text = list(df.ix[:, 0])
    text = [txt.lower() for txt in text]
    for txt in text:
        words = words + re.findall(r'\w+', txt)

    return words


# Create word count dictionary
def get_count(word_l):
    """
    Input:
        word_l: a set of words representing the corpus.
    Output:
        word_count_dict: The wordcount dictionary where key is the word and value is its frequency.
    """

    word_count_dict = {}  # fill this with word counts

    for w in word_l:
        word_count_dict[w] = word_count_dict.get(w, 0) + 1

    return word_count_dict


# Get words apparition probabilities
def get_probs(word_count_dict):
    """
    Input:
        word_count_dict: The wordcount dictionary where key is the word and value is its frequency.
    Output:
        probs: A dictionary where keys are the words and the values are the probability that a word will occur.
    """
    probs = {}  # return this variable correctly

    # Compute total number of words
    total_nb_words = 0
    for w in word_count_dict:
        total_nb_words = total_nb_words + word_count_dict.get(w, 0)

    for w in word_count_dict:
        probs[w] = word_count_dict.get(w, 0) / total_nb_words
    return probs


# Delete, switch, insert
def delete_letter(word, verbose=False):
    """
    Input:
        word: the string/word for which you will generate all possible words
                in the vocabulary which have 1 missing character
    Output:
        delete_l: a list of all possible strings obtained by deleting 1 character from word
    """

    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    delete_l = [L + R[1:] for L, R in split_l if R]

    if verbose: print(f"input word {word}, \nsplit_l = {split_l}, \ndelete_l = {delete_l}")

    return delete_l


def switch_letter(word, verbose=False):
    """
    Input:
        word: input string
     Output:
        switches: a list of all possible strings with one adjacent charater switched
    """
    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    switch_l = [L[:(len(L) - 1)] + R[0] + L[(len(L) - 1)] + R[1:] for L, R in split_l if R and L]

    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nswitch_l = {switch_l}")

    return switch_l


def replace_letter(word, verbose=False):
    """
    Input:
        word: the input string/word
    Output:
        replaces: a list of all possible strings where we replaced one letter from the original word.
    """

    letters = 'abcdefghijklmnopqrstuvwxyz'
    split_l = [(word[:i], word[i:]) for i in range(len(word))]
    replace_l_1 = [L[:(len(L) - 1)] + letter + R for L, R in split_l if L and R for letter in letters]
    replace_l_2 = [L + letter for L, R in split_l if L and R and len(R) == 1 for letter in letters]

    replace_l = replace_l_1 + replace_l_2
    replace_l = [wrd for wrd in replace_l if wrd != word]

    # turn the set back into a list and sort it, for easier viewing
    replace_l = sorted(list(replace_l))

    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nreplace_l {replace_l}")

    return replace_l


def insert_letter(word, verbose=False):
    """
    Input:
        word: the input string/word
    Output:
        inserts: a set of all possible strings with one new letter inserted at every offset
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'

    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    insert_l = [L + letter + R for L, R in split_l for letter in letters]

    if verbose: print(f"Input word {word} \nsplit_l = {split_l} \ninsert_l = {insert_l}")

    return insert_l


# Combining editions
def edit_one_letter(word, allow_switches=True):
    """
    Input:
        word: the string/word for which we will generate all possible wordsthat are one edit away.
    Output:
        edit_one_set: a set of words with one possible edit. Please return a set. and not a list.
    """

    res = delete_letter(word) + replace_letter(word) + insert_letter(word)
    if allow_switches:
        res = res + switch_letter(word)

    edit_one_set = set(res)

    return edit_one_set


def edit_two_letters(word, allow_switches=True):
    """
    Input:
        word: the input string/word
    Output:
        edit_two_set: a set of strings with all possible two edits
    """

    edit_two_set = []
    for wd in edit_one_letter(word, allow_switches):
        edit_two_set = edit_two_set + list(edit_one_letter(wd, allow_switches))

    return set(edit_two_set)


# Spelling suggestions
def get_corrections(word, probs, vocab, n=2, verbose=False):
    """
    Input:
        word: a user entered string to check for suggestions
        probs: a dictionary that maps each word to its probability in the corpus
        vocab: a set containing all the vocabulary
        n: number of possible word corrections you want returned in the dictionary
    Output:
        n_best: a list of tuples with the most probable n corrected words and their probabilities.
    """

    suggestions = [wd for wd in vocab if wd == word] or [wd for wd in edit_one_letter(word) if wd in vocab] or [wd for wd in edit_two_letters(word) if wd in vocab] or word
    suggestions = sorted(suggestions)
    sorted_probs = {k: v for k, v in sorted(probs.items(), key=lambda item: item[1])}
    sorted_probs = {x: sorted_probs[x] for x in suggestions}
    n_best = list(sorted_probs.items())[:n]

    if verbose: print("entered word = ", word, "\nsuggestions = ", suggestions)

    return n_best


# Minimum edit distance function
def min_edit_distance(source, target, ins_cost=1, del_cost=1, rep_cost=2):
    """
    Input:
        source: a string corresponding to the string you are starting with
        target: a string corresponding to the string you want to end with
        ins_cost: an integer setting the insert cost
        del_cost: an integer setting the delete cost
        rep_cost: an integer setting the replace cost
    Output:
        D: a matrix of len(source)+1 by len(target)+1 containing minimum edit distances
        med: the minimum edit distance (med) required to convert the source string to the target
    """
    # use deletion and insert cost as  1
    m = len(source)
    n = len(target)
    # initialize cost matrix with zeros and dimensions (m+1,n+1)
    D = np.zeros((m + 1, n + 1), dtype=int)

    # Fill in column 0, from row 1 to row m, both inclusive
    for row in range(0, m + 1):  # Replace None with the proper range
        D[row, 0] = row

    # Fill in row 0, for all columns from 1 to n, both inclusive
    for col in range(0, n + 1):  # Replace None with the proper range
        D[0, col] = col

    # Loop through row 1 to row m, both inclusive
    for row in range(1, m + 1):

        # Loop through column 1 to column n, both inclusive
        for col in range(1, n + 1):

            # Intialize r_cost to the 'replace' cost that is passed into this function
            r_cost = rep_cost

            # Check to see if source character at the previous row
            # matches the target character at the previous column,
            if source[row - 1] == target[col - 1]:
                # Update the replacement cost to 0 if source and target are the same
                r_cost = 0

            # Update the cost at row, col based on previous entries in the cost matrix
            # Refer to the equation calculate for D[i,j] (the minimum of three calculated costs)
            D[row, col] = min([D[row - 1, col] + del_cost, D[row, col - 1] + ins_cost, D[row - 1, col - 1] + r_cost])

    # Set the minimum edit distance with the cost found at row m, column n
    med = D[m, n]
    return D, med