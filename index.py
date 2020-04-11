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
        set_of_documents = self.get_documents()
        # Save flattened Counter results in tokens_list
        for res in set_of_documents:
            doc_id = res['doc_id']
            positional_indexes = res['positional_indexes']
            for term, positions in positional_indexes.items():
                tokens_list.append([term, (doc_id, positions, res['title'], res['court'])])
        tokens_list.sort(key=functools.cmp_to_key(comparator)) # Sorted list of [term, (doc_id, freq_in_doc)] elements

        # Step 2: Get a list of all available doc_ids in ascending order
        self.doc_ids = sorted(list(set([el['doc_id'] for el in set_of_documents])))

        print("Generating posting lists")

        # Step 3: Fill up the dictionary with PostingLists of all unique terms
        # The dictionary maps the term to its PostingList
        for i in range(len(tokens_list)):
            curr = tokens_list[i]
            term = curr[0]
            curr_tuple = curr[1] # (doc_id, term frequency, title, court)
            if i == 0 or term != tokens_list[i-1][0]:
                # new term
                self.dictionary[term] = PostingList()
            # insert into appropriate PostingList
            self.dictionary[term].insert(curr_tuple[0], curr_tuple[1], curr_tuple[2], curr_tuple[3])

        print("Calculating document vector length")

        # Step 4: Calculate doc_lengths for normalization
        self.calculate_doc_length()

        for _, posting_list in self.dictionary.items():
            posting_list.generate_skip_list()

    # Read the file in and split by the characters '",'
    # Processes the data set into an array of cases/documents
    # Note: This function DOES NOT filter punctuation and lower case
    def process_file(self):
        with open(self.in_dir, encoding='utf-8') as f:
            # prevent csv field larger than field limit error
            csv.field_size_limit(sys.maxsize)

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
                  document['title'] = row[3].strip('')
                  document['court'] = row[4].strip('')
                  documents.append(document)
                index += 1
                # TODO: Delete
                # if index == 600:
                  # break
            return documents

    def get_documents(self):
        # Result container for collating all possible dictionary file terms
        set_of_documents = []
        documents = self.process_file()
        print("Done processing file")

        # Then we handle the content
        # For zones, we adopt a broad zoning procedure, with Judgment and non Judgment being the only zone
        count = 0
        for document in documents:
            sentences = nltk.sent_tokenize(document['content'])
            words_array = [nltk.word_tokenize(s) for s in sentences]
            words = [w for arr in words_array for w in arr]
            processed_words = self.process_words(words)
            # This is a dictionary of word and the positions it appears in
            positional_indexes = self.generate_positional_indexes(processed_words, 0)
            document['positional_indexes'] = positional_indexes
            set_of_documents.append(document)
            print(count," Generated poisitional index")
            count += 1

            # Lower case judgement here because process_file() already lowercased everything
            # zones = document.content.split("judgment:")
            # non_judgment_zone, judgment_zone = zones[0], zones[1]

            # # Do for non judgment zone first
            # # Consider abstracting out into a new function of we have more zones
            # non_judgement_sentences = nltk.sent_tokenize(non_judgment_zone)
            # non_judgement_words_array = [nltk.word_tokenize(s) for s in non_judgement_sentences]
            # non_judgement_words = [w for arr in non_judgement_words_array for w in arr]
            # non_judgement_processed_words = self.process_words(non_judgement_words)
            # # This is a dictionary of word and the positions it appears in
            # non_judgement_positional_indexes = self.generate_positional_indexes(non_judgement_processed_words, 0)
            # non_judgement_length = len(non_judgement_processed_words)

            # judgement_sentences = nltk.sent_tokenize(judgment_zone)
            # judgement_words_array = [nltk.word_tokenize(s) for s in judgement_sentences]
            # judgement_words = [w for arr in judgement_words_array for w in arr]
            # judgement_processed_words = self.process_words(judgement_words)
            # # This is a dictionary of word and the positions it appears in
            # judgement_positional_indexes = self.generate_positional_indexes(judgement_processed_words, non_judgement_length)

            # document.non_judgement_indexes = non_judgement_positional_indexes
            # document.judgment_indexes = judgement_positional_indexes
        print("Done getting documents")

        return set_of_documents

    # This function aims to generate the positional indexes for the phrasal queries
    def generate_positional_indexes(self, words, start_index):
        positions = defaultdict(list)
        last_position = {}
        for i in range(start_index, len(words)):
            word = words[i]
            if word not in last_position:
              last_position[word] = i
              positions[word].append(i)
            else:
              positions[word].append(i - last_position[word])
              last_position[word] = i
        return positions

    def process_words(self, words):
        """
        Stems the already lowercase version of the word given and lowercases
        Takes in the list of Strings and returns their stemmed version in a list
        """
        stemmer = nltk.stem.porter.PorterStemmer()
        return [stemmer.stem(w.lower()) for w in words]

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
                    posting_weight = 1 + math.log(len(posting.positions), 10)
                    self.doc_lengths[posting.doc_id] = posting_weight * posting_weight
                else:
                    self.doc_lengths[posting.doc_id] += (posting_weight * posting_weight)
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
            # pickle.dump(self.doc_ids, f)
            for word, posting_list in self.dictionary.items():
                cursor = f.tell()
                d[word] = cursor # updating respective (term to file cursor value) mappings
                # pickle.dump([posting.to_dict() for posting in posting_list.postings], f, protocol=4)
                pickle.dump(posting_list, f, protocol=4)

        with open(self.d_file, "wb") as f:
            pickle.dump(d, f) # (term to file cursor value) mappings dictionary
            pickle.dump(self.doc_lengths, f)

class Posting:
    """
    Each Posting has a document id doc_id, term frequency freq,
    and weight which is its lnc calculation before normalisation
    """
    def __init__(self, index, doc_id, positions, title, court):
        # self.index = index # 0 indexed
        self.doc_id = doc_id
        # self.freq = len(positions) # term frequency of the term in that document
        # self.weight = 1 + math.log(len(positions), 10) # lnc calculation before normalisation (done during search)
        self.positions = positions
        self.pointer = None
        # self.title = title
        # self.court = court

class PostingList:
    """
    Each PostingList is for a term, reflecting its term frequency in a number of document(s)
    A PostingList contains Postings, which have doc_id and freq of each term
    """
    def __init__(self):
        self.postings = []
        self.size = 0
        self.last = 0

    def get_size(self):
        return self.size

    def insert(self, doc_id, positions, title, court):
        next_id = self.size
        if next_id == 0:
          self.postings.append(Posting(next_id, doc_id, positions, title, court))
        else:
          self.postings.append(Posting(next_id, doc_id - self.last, positions, title, court))
        self.size += 1
        self.last = doc_id

    def insert_posting(self, posting):
        self.postings.append(posting)

    def get(self, index):
        return self.postings[index]

    def generate_skip_list(self):
        skip_distance = math.floor(math.sqrt(self.get_size()))
        # -1 to prevent reading from invalid index
        for i in range(self.get_size() - skip_distance - 1):
            self.postings[i].pointer = self.postings[i + skip_distance]

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
