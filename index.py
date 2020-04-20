#!/usr/bin/python3
import re
import sys
import getopt
import math
import os
import nltk
import pickle
import csv
import functools
from collections import Counter, defaultdict
from encode import encode
from enum import IntEnum

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
    Sorts the 2 lists by term first, then doc_id in ascending order, then field (title -> court -> content)
    """
    if arr1[0] > arr2[0]:  # by term
        return 1
    elif arr2[0] > arr1[0]:
        return -1
    elif arr1[1][0] != arr2[1][0]:  # same term and different doc_id  -> by doc_id
        return arr1[1][0] - arr2[1][0]
    elif arr1[1][1] != arr2[1][1]:
        # same term and doc_id: they can only differ by: title -> court -> content
        # note the following cases:
        # both are title - does not matter which is placed first
        # one is of higher priority - place the higher priority one first

        # title
        if arr2[1][1] == Field.TITLE:
            return 1
        elif arr1[1][1] == Field.TITLE:
            return -1

        # court
        elif arr2[1][1] == Field.COURT:
            return 1
        elif arr1[1][1] == Field.COURT:
            return -1

    else:
        # content
        return 0

class VSM:
    """
    Represents the Vector Space Model
    """
    def __init__(self, in_dir, d_file, p_file):
        self.dictionary = {}  # content, title, court
        self.docid_set = set() # docID
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

        # We obtain a list of documents with keys:
        # 'doc_id', 'title', 'content', 'date_posted', 'court', as well as positional indexes of
        # 'content_positional_indexes', 'title_positional_indexes', 'court_positional_indexes'
        set_of_documents = self.get_documents()
        tokens_list = []
        # Save flattened Counter results in tokens_list
        for res in set_of_documents:
            doc_id = res['doc_id']
            # Generate tokens of positional indexes (subsequently form into posting list)
            tokens_list.extend(self.generate_token_list(doc_id, Field.CONTENT, res['content_positional_indexes']))
            tokens_list.extend(self.generate_token_list(doc_id, Field.TITLE, res['title_positional_indexes']))
            tokens_list.extend(self.generate_token_list(doc_id, Field.COURT, res['court_positional_indexes']))

        tokens_list.sort(key=functools.cmp_to_key(comparator)) # Sorted list of [term, (doc_id, freq_in_doc)] elements

        # tokens_list is now a list of structured data that has all terms sorted by their values:
        # firstly by term, then doc_id, then title, court, content
        # note that actual positiions are gap encoded, so only the very first entry is the actual position;
        # the rest need to be calculated incrementally

        # Step 2: Get a list of all available doc_ids in ascending order
        self.docid_set = set([el['doc_id'] for el in set_of_documents])  # Process doc_id into a set

        print("Generating posting lists")

        # Step 3: Fill up the dictionary with PostingLists of all unique terms
        # The dictionary maps the term to its PostingList
        for i in range(len(tokens_list)):

            curr = tokens_list[i]
            term = curr[0]
            curr_tuple = curr[1] # (doc_id, Field, freq_in_doc)

            if (i == 0 or term != tokens_list[i-1][0]):
                # new term detected, create new PostingList
                self.dictionary[term] = PostingList()

            # insert into appropriate PostingList
            self.dictionary[term].insert(curr_tuple[0], curr_tuple[1], curr_tuple[2])

        # Step 4: Calculate doc_lengths for normalization
        print("Calculating document vector length")
        self.calculate_doc_length()

        for _, posting_list in self.dictionary.items():
            posting_list.generate_skip_list()

    def get_documents(self):
        """
        Returns a list of documents (each document is represented by a dictionary)
        Dictionary/dictionary keys:
        'doc_id', 'title', 'content', 'date_posted', 'court', as well as positional indexes of
        'content_positional_indexes', 'title_positional_indexes', 'court_positional_indexes'
        """
        # Result container for collating all possible dictionary file terms
        set_of_documents = []

        # A document is one part of the csv file that has:
        # doc_id, title, content, date_posted, court
        # Each of these are keys to access using a dictionary
        documents = self.process_file() # split the csv file to get all documents
        print("Done processing file")

        # Create 'content_positional_indexes', 'title_positional_indexes', 'court_positional_indexes'
        # Each of these are specific to a document
        # These are in addition to the (doc_id, title, content, date_posted, court) previously
        # Each document is updated with these new indexes, are stored in set_of_documents
        count = 0
        for document in documents:
            document['content_positional_indexes'] = self.generate_positional_indexes(document['content'])  # Part 1: Content
            document['title_positional_indexes'] = self.generate_positional_indexes(document['title'])  # Part 2: Title
            document['court_positional_indexes'] = self.generate_positional_indexes(document['court'])  # Part 3: Court
            set_of_documents.append(document)

            print(count," Generated positional indexes")
            count += 1
        print("Done getting documents")

        return set_of_documents

    # Read the file in and split by the characters '",'
    # Processes the data set into an array of cases/documents
    # Note: This function DOES NOT filter punctuation and lower case
    def process_file(self):
        """
        Reads in the dataset file and returns all individual documents in a list
        """
        with open(self.in_dir, encoding='utf-8') as f:
            """
            # prevent csv field larger than field limit error
            csv.field_size_limit(sys.maxsize)
            """
            # # TODO: Delete?
            # To run csv.field_size_limit(sys.maxsize) LOCALLY
            # by resolving "OverflowError: Python int too large to convert to C long"
            # prevent csv field larger than field limit error
            maxInt = sys.maxsize
            while True:
                # decrease the maxInt value by factor 10
                # as long as the OverflowError occurs.
                try:
                    csv.field_size_limit(maxInt)
                    break
                except OverflowError:
                    maxInt = int(maxInt / 10)

            csv_reader = csv.reader(f, delimiter=',')
            documents = []
            index = 0
            for row in csv_reader:
                if index != 0:
                  document = {}
                  # Renaming columns here so we cant use csv.DictReader
                  document['doc_id'] = int(row[0].strip(''))
                  document['title'] = row[1].strip('')
                  document['content'] = row[2].strip('')
                  document['date_posted'] = row[3].strip('')
                  document['court'] = row[4].strip('')
                  documents.append(document)
                index += 1

"""
                # # TODO: Delete - THIS IS FOR TESTING
                if index == 60:
                  break
"""

            return documents

    def generate_positional_indexes(self, paragraph):
        """
        Generates positional index from a paragraph/string
        by using the function generate_list_of_words() and generate_positional_indexes_from_list()
        """
        processed_words = self.generate_list_of_words(paragraph)
        # This is a dictionary of word and the (gap-encoded) positions it appears in
        positional_indexes = self.generate_positional_indexes_from_list(processed_words, 0)
        return positional_indexes

    def generate_list_of_words(self, paragraph):
        """
        Generates a list of processed words from the string it was input with
        """
        # break into word-level terms
        sentences = nltk.sent_tokenize(paragraph)
        words_array = [nltk.word_tokenize(s) for s in sentences]
        words = [w for arr in words_array for w in arr]
        # preprocess individual words
        processed_words = self.process_words(words)
        return processed_words

    def process_words(self, words):
        """
        Stems the words given and casefolds them to lowercase
        Takes in the list of Strings and returns their stemmed version in a list
        """
        stemmer = nltk.stem.porter.PorterStemmer()
        return [stemmer.stem(w.lower()) for w in words]

    def generate_positional_indexes_from_list(self, words, start_index):
        """
        Generate the positional indexes from a list of words (applying gap encoding for positions)
        This is used for phrasal queries during query phase
        Therefore, compression is applied here via gap encoding
        """
        # positions is a dictionary scoring all the terms and their positions in a list each
        # Each term has their own list of positions
        positions = defaultdict(list)
        # last_position is a dictionary storing each respective word's last seen position
        # This is not a list
        # This is used for calculating the gap encoding values by substraction
        last_position = {}

        for i in range(start_index, len(words)):
            word = words[i]
            # Store gap encoding
            if word not in last_position:
                # totally new word, has no previous occurences
                last_position[word] = i # initilaise its position list with this position
                positions[word].append(i) # add this position to the list
            else:
                # word is seen before
                positions[word].append(i - last_position[word]) # update the position list with the gap
                last_position[word] = i # update the last seen position

        return positions

    def generate_token_list(self, doc_id, field_type, positional_indexes):
        """
        Generates token from a list of positional index
        """
        tokens_list = []
        for term, position in positional_indexes.items():
            tokens_list.append([term, (doc_id, field_type, position)])  # [term, (doc_ID, Field, position)]
        return tokens_list

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
                posting_weight = 1 + math.log(len(posting.positions), 10)
                if posting.doc_id not in self.doc_lengths:
                    self.doc_lengths[posting.doc_id] = posting_weight * posting_weight
                else:
                    self.doc_lengths[posting.doc_id] += (posting_weight * posting_weight)
        for doc_id, total_weight in self.doc_lengths.items():
            self.doc_lengths[doc_id] = math.sqrt(total_weight)

    def write(self):
        """
        Writes PostingList objects into postings file and all terms into dictionary file
        Document lengths are also written into dictionary file
        All doc_ids are also written into dictionary file
        """
        #out_intermediate = open("intermediate.txt", "w", encoding='utf8')  # For debuging purpose, TO DELETE

        d = {}  # to contain mappings of term to file cursor value
        with open(self.p_file, "wb") as f:
            # pickle.dump(self.doc_ids, f)
            for word, posting_list in self.dictionary.items():
                cursor = f.tell()
                d[word] = cursor # updating respective (term to file cursor value) mappings
                pickle.dump(posting_list, f, protocol=4)
                #out_intermediate.write(
                #    "Word: " + str(word) + " Posting: " + posting_list.generate_string_of_postinglist() + '\n\n')  # For debuging purpose, TO DELETE

        with open(self.d_file, "wb") as f:
            pickle.dump(d, f) # (term to file cursor value) mappings dictionary
            pickle.dump(self.doc_lengths, f)
            pickle.dump(self.docid_set, f)

            #out_intermediate.write("dictionary: " + str(d) + '\n\n')  # For debuging purpose, TO DELETE
            #out_intermediate.write("docid_set: " + str(self.docid_set) + '\n\n')  # For debuging purpose, TO DELETE

        #out_intermediate.close()  # For debuging purpose, TO DELETE

class Field(IntEnum):
    CONTENT = 1
    TITLE = 2
    COURT = 3

class Posting:
    """
    Each Posting has a document id doc_id, term frequency freq,
    and weight which is its lnc calculation before normalisation
    """
    def __init__(self, index, doc_id, field, positions):
        self.doc_id = doc_id
        self.field = field
        self.positions = positions
        self.pointer = None

    def var_byte_encoding(self):
        self.positions = encode(self.positions)

    def generate_string_of_posting(self):
        return ' (' + str(self.doc_id) + ', ' + str(self.field) + ', ' + str(self.positions) + ') '

class PostingList:
    """
    Each PostingList is for a term, reflecting its term frequency in a number of document(s)
    A PostingList contains size and Postings, which have doc_id and freq of each term
    """
    def __init__(self):
        self.postings = []
        self.size = 0  # number of insert / term frequency

    def get_size(self):
        return self.size

    # Insert with var byte encoding
    def insert(self, doc_id, field, positions):
        next_id = self.size
        new_posting = Posting(next_id, doc_id, field, positions)
        new_posting.var_byte_encoding()
        self.postings.append(new_posting)
        self.size += 1

    def insert_without_encoding(self, doc_id, field, positions):
        next_id = self.size
        self.postings.append(Posting(next_id, doc_id, field, positions))
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

    def generate_string_of_postinglist(self):
        s = 'size ' + str(self.size) + '  '
        for item in self.postings:
            s += item.generate_string_of_posting() + ';'
        return s

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
