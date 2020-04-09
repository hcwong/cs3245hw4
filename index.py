#!/usr/bin/python3
import re
import sys
import getopt
import math
import os
import nltk
import pickle
import functools
from collections import Counter, defaultdict

# Self-defined constants, functions and classes

def filter_punctuations(s):
    """
    Replaces certain punctuations from Strings with space, to be removed later on
    Takes in String s and returns the processed version of it
    """
    punctuations = '''!?-;:"\\,./#$%^&<>[]{}*_~()'''
    for character in s:
        if character in punctuations:
            s = s.replace(character, " ")
    return s

def comparator(arr1, arr2):
    """
    Sorts the 2 lists by term first, then doc_id in ascending order
    """
    if arr1[0] > arr2[0]:
        return 1
    elif arr2[0] > arr1[0]:
        return -1
    else:
        return arr1[1][0] - arr2[1][0]

class VSM:
    """
    Represents the Vector Space Model
    """
    def __init__(self, in_dir, d_file, p_file):
        self.dictionary = {}
        self.in_dir = in_dir
        self.d_file = d_file
        self.p_file = p_file

    def build(self):
        """
        Builds the Vector Space Model (VSM), which includes a dictionary of PostingLists for each term
        A dictionary of document lengths and a list of document ids are also made
        These are accessed via .dictionary, .doc_lengths, .doc_ids respectively
        Punctuation handling, tokenisation, case-folding, stemming are applied to generate terms
        """
        # Step 1: Obtain Postings for material to create VSM in Step 3
        tokens_list = []
        set_of_words = self.get_words()
        # Save flattened Counter results in tokens_list
        for res in set_of_words:
            doc_id = res[0]
            # This is a dict of [word, [positions]]
            positional_indexes = res[1]
            for term, positions in positional_indexes.items():
                tokens_list.append([term, (doc_id, positions)])
        tokens_list.sort(key=functools.cmp_to_key(comparator)) # Sorted list of [term, (doc_id, freq_in_doc)] elements

        # Step 2: Get a list of all available doc_ids in ascending order
        self.doc_ids = sorted(list(set([el[0] for el in set_of_words])))

        # Step 3: Fill up the dictionary with PostingLists of all unique terms
        # The dictionary maps the term to its PostingList
        for i in range(len(tokens_list)):
            curr = tokens_list[i]
            term = curr[0]
            curr_tuple = curr[1] # (doc_id, term frequency)
            if i == 0 or term != tokens_list[i-1][0]:
                # new term
                self.dictionary[term] = PostingList()
            # insert into appropriate PostingList
            self.dictionary[term].insert(curr_tuple[0], curr_tuple[1])

        # Step 4: Calculate doc_lengths for normalization
        self.calculate_doc_length()

        for _, posting_list in self.dictionary.items():
            posting_list.generate_skip_list()

    def get_words(self):
        """
        Obtains a list of tuples (doc_id, Counter(all of doc_id's {term: term frequency} entries))
        to construct the VSM, constructed from all the files in the directory
        Returns all possible processed terms in a list of (doc_id, Counter)
        Punctuation handling, case-folding, tokenisation, and stemming are applied
        """
        # Result container for collating all possible dictionary file terms
        set_of_words = []

        for filename in os.listdir(self.in_dir):

            # Container for all words within a single file
            words = []

            with open(f"{self.in_dir}/{filename}", encoding="utf8") as f:
                # Here, we assume that it is okay to load everything into memory
                # Step 1: Read in entire file, filtering out relevant punctuations
                # and replaces hyphens with spaces
                words = filter_punctuations(f.read()).lower()
                # Step 2: Obtaining terms and tokenising the content of the file
                sentences = nltk.sent_tokenize(words)
                words_array = [nltk.word_tokenize(s) for s in sentences]
                words = [w for arr in words_array for w in arr]
                processed_words = self.process_words(words) # may contain duplicate terms
                # Step 3: Accumulate all tuples of (doc_id, Counter(all of doc_id's {term: term frequency} entries))
                positional_indexes = self.generate_positional_indexes(processed_words)
                set_of_words.append((int(filename), positional_indexes))

        return set_of_words

    # This function aims to generate the positional indexes for the phrasal queries
    def generate_positional_indexes(self, words):
        positions = defaultdict(list)
        for i in range(len(words)):
            word = words[i]
            positions[word].append(i)
        return positions

    def process_words(self, words):
        """
        Stems the already lowercase version of the word given
        Takes in the list of Strings and returns their stemmed version in a list
        """
        stemmer = nltk.stem.porter.PorterStemmer()
        return [stemmer.stem(w) for w in words]

    def calculate_doc_length(self):
        """
        Sets and stores the length of each document for normalization
        """
        self.doc_lengths = {}
        # Iterate through every term of the dictionary, getting its PostingList
        # to iterate every posting and calculate its contribution to its document's vector length
        # then, complete calculation for the vector's length
        for _, posting_list in self.dictionary.items():
            for posting in posting_list.postings:
                if posting.doc_id not in self.doc_lengths:
                    self.doc_lengths[posting.doc_id] = posting.weight * posting.weight
                else:
                    self.doc_lengths[posting.doc_id] += (posting.weight * posting.weight)
        for doc_id, total_weight in self.doc_lengths.items():
            self.doc_lengths[doc_id] = math.sqrt(total_weight)

    def write(self):
        """
        Writes PostingList objects into postings file and all terms into dictionary file
        Document lengths are also written into dictionary file
        All doc_ids are also written into postings file
        """
        d = {} # to contain mappings of term to file cursor value
        with open(self.p_file, "wb") as f:
            pickle.dump(self.doc_ids, f)
            for word, posting_list in self.dictionary.items():
                cursor = f.tell()
                d[word] = cursor # updating respective (term to file cursor value) mappings
                pickle.dump(posting_list, f)

        with open(self.d_file, "wb") as f:
            pickle.dump(d, f) # (term to file cursor value) mappings dictionary
            pickle.dump(self.doc_lengths, f)

class Posting:
    """
    Each Posting has a document id doc_id, term frequency freq,
    and weight which is its lnc calculation before normalisation
    """
    def __init__(self, index, doc_id, positions):
        self.index = index # 0 indexed
        self.doc_id = doc_id
        self.freq = len(positions) # term frequency of the term in that document
        self.weight = 1 + math.log(len(positions), 10) # lnc calculation before normalisation (done during search)
        self.positions = positions
        self.pointer = None

class PostingList:
    """
    Each PostingList is for a term, reflecting its term frequency in a number of document(s)
    A PostingList contains Postings, which have doc_id and freq of each term
    """
    def __init__(self):
        self.postings = []
        self.size = 0

    def get_size(self):
        return self.size

    def insert(self, doc_id, positions):
        # Creates a new Posting and places it at the next available location,
        # leaving no spaces (compact)
        next_id = self.size
        self.postings.append(Posting(next_id, doc_id, positions))
        self.size += 1

    def insert_posting(self, posting):
        self.postings.append(posting)

    def get(self, index):
        return self.postings[index]

    def generate_skip_list(self):
        skip_distance = math.floor(math.sqrt(self.get_size()))
        # -1 to prevent reading from invalid index
        for i in range(self.get_size() - skip_distance - 1):
            self.postings[i].pointer = self.postings[i + skip_distance]

# Below are the code provided in the original Homework index.py file,
# with edits to build_index to use our implementation

def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")

def build_index(in_dir, out_dict, out_postings):
    """
    Build index from documents stored in the input directory
    then output the dictionary file and postings file
    """
    print('indexing...')
    vsm = VSM(in_dir, out_dict, out_postings)
    vsm.build()
    vsm.write()

input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input directory
        input_directory = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
