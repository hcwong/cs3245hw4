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

def filter_punctuations(s, keep_quo=False):
    """
    Replaces certain punctuations from Strings with space, to be removed later on
    Takes in String s and returns the processed version of it
    """
    punct_wo_quo = '''!?-;:\\,./#$%^&<>[]{}*`'=@+…’-–—_~()'''
    punctuations = '''!?-;:"\\,./#$%^&<>[]{}*`'=@+…“”’-–—_~()'''

    if keep_quo:
        for character in s:
            if character in punct_wo_quo:
                s = s.replace(character, " ")
    else:
        for character in s:
            if character in punctuations:
                s = s.replace(character, " ")
    return s

def comparator(arr1, arr2):
    """
    Sorts the 2 lists by term first, then doc_id in ascending order, then field (title -> court -> date_posted-> content)
    """
    if arr1[0] > arr2[0]:  # term
        return 1
    elif arr2[0] > arr1[0]:
        return -1
    elif arr1[1][0] != arr2[1][0]:  # doc_id
        return arr1[1][0] - arr2[1][0]
    elif arr1[1][1] != arr2[1][1]:  # Field title -> court -> content
        if arr2[1][1] == Field.TITLE:
            return 1
        elif arr1[1][1] == Field.TITLE:
            return -1
        elif arr2[1][1] == Field.COURT:
            return 1
        elif arr1[1][1] == Field.COURT:
            return -1
        elif arr2[1][1] == Field.DATE_POSTED:
            return 1
        elif arr1[1][1] == Field.DATE_POSTED:
            return -1
    else:
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
        set_of_documents = self.get_documents()
        tokens_list = []

        # Save flattened Counter results in tokens_list
        for res in set_of_documents:
            doc_id = res['doc_id']
            # Generate tokens of positional indexes (subsequently form into posting list)
            tokens_list.extend(self.generate_token_list(doc_id, Field.CONTENT, res['content_positional_indexes']))
            tokens_list.extend(self.generate_token_list(doc_id, Field.TITLE, res['title_positional_indexes']))
            tokens_list.extend(self.generate_token_list(doc_id, Field.COURT, res['court_positional_indexes']))
            tokens_list.extend(self.generate_token_list(doc_id, Field.DATE_POSTED, res['date_posted_positional_indexes']))

        tokens_list.sort(key=functools.cmp_to_key(comparator)) # Sorted list of [term, (doc_id, freq_in_doc)] elements

        # Step 2: Get a list of all available doc_ids in ascending order
        self.docid_set = set([el['doc_id'] for el in set_of_documents])  # Process doc_id into a set

        print("Generating posting lists")

        # Step 3: Fill up the dictionary with PostingLists of all unique terms
        # The dictionary maps the term to its PostingList
        for i in range(len(tokens_list)):
            curr = tokens_list[i]
            term = curr[0]
            curr_tuple = curr[1] # (doc_id, Field, freq_in_doc)
            if i == 0 or term != tokens_list[i-1][0]:
                # new term
                self.dictionary[term] = PostingList()
            # Insert into appropriate PostingList
            # if same term and docID, do not increment PL.size
            if i > 0 and term == tokens_list[i-1][0] and curr_tuple[0] == tokens_list[i-1][1][0]:
                self.dictionary[term].insert(curr_tuple[0], curr_tuple[1], curr_tuple[2], False)
            else:
                self.dictionary[term].insert(curr_tuple[0], curr_tuple[1], curr_tuple[2])

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
                # # TODO: Delete
                if index == 60:
                    break
            return documents

    def get_documents(self):
        # Result container for collating all possible dictionary file terms
        set_of_documents = []
        documents = self.process_file()
        print("Done processing file")

        # Then we handle the fields: content, title and court (to generate position index)
        # of 'content_positional_indexes', 'title_positional_indexes' is made from 'title', 'court_positional_indexes'
        count = 0
        for document in documents:
            document['content_positional_indexes'] = self.generate_positional_indexes(document['content'])  # Part 1: Content
            document['title_positional_indexes'] = self.generate_positional_indexes(document['title'])  # Part 2: Title
            document['court_positional_indexes'] = self.generate_positional_indexes(document['court'])  # Part 3: Court
            document['date_posted_positional_indexes'] = self.generate_positional_indexes(document['date_posted'].split()[0])  # Part 4: Date_posted
            set_of_documents.append(document)
            print(count," Generated positional indexes")
            count += 1

        print("Done getting documents")
        return set_of_documents

    def generate_token_list(self, doc_id, field_type, positional_indexes):
        """
        Generates token from a list of positional index
        """
        tokens_list = []
        for term, positions in positional_indexes.items():
            tokens_list.append([term, (doc_id, field_type, positions)])  # [term, (doc_ID, Field, position]
        return tokens_list

    def generate_positional_indexes(self, paragraph):
        """
        #Generates positional index from a paragraph/string
        #by using the function generate_list_of_words() and generate_positional_indexes_from_list()
        """
        processed_words = self.generate_list_of_words(paragraph)
        # This is a dictionary of word and the positions it appears in
        positional_indexes = self.generate_positional_indexes_from_list(processed_words, 0)
        return positional_indexes

    def generate_list_of_words(self, paragraph):
        """
        Generates a list of processed words from the string it was input with
        """
        sentences = nltk.sent_tokenize(paragraph)
        words_array = [nltk.word_tokenize(filter_punctuations(s)) for s in sentences]
        words = [w for arr in words_array for w in arr]
        processed_words = self.process_words(words)
        return processed_words

    def generate_positional_indexes_from_list(self, words, start_index):
        """
        Generate the positional indexes for the phrasal queries from a list of words
        Does gap encoding too
        """
        positions = defaultdict(list)
        last_position = {}
        for i in range(start_index, len(words)):
            word = words[i]
            # Store gap encoding
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
        All doc_ids are also written into postings file
        """
        out_intermediate = open("intermediate.txt", "w", encoding='utf8')  # For debuging purpose, TO DELETE

        d = {}  # to contain mappings of term to file cursor value
        with open(self.p_file, "wb") as f:
            # pickle.dump(self.doc_ids, f)
            for word, posting_list in self.dictionary.items():
                cursor = f.tell()
                d[word] = cursor # updating respective (term to file cursor value) mappings
                pickle.dump(posting_list, f, protocol=4)
                out_intermediate.write(
                    "Word: " + str(word) + " Posting: " + posting_list.generate_string_of_postinglist() + '\n\n')  # For debuging purpose, TO DELETE

        with open(self.d_file, "wb") as f:
            pickle.dump(d, f) # (term to file cursor value) mappings dictionary
            pickle.dump(self.doc_lengths, f)
            pickle.dump(self.docid_set, f)

            out_intermediate.write("dictionary: " + str(d) + '\n\n')  # For debuging purpose, TO DELETE
            out_intermediate.write("docid_set: " + str(self.docid_set) + '\n\n')  # For debuging purpose, TO DELETE
        out_intermediate.close()  # For debuging purpose, TO DELETE

class Field(IntEnum):
    CONTENT = 1
    TITLE = 2
    COURT = 3
    DATE_POSTED = 4

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
    def insert(self, doc_id, field, positions, new_doc_id=True):
        next_id = self.size
        new_posting = Posting(next_id, doc_id, field, positions)
        new_posting.var_byte_encoding()
        self.postings.append(new_posting)
        if new_doc_id:
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

if __name__ == "__main__":
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
