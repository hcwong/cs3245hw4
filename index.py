#!/usr/bin/python3
# -*- coding: utf-8 -*-

import re
import sys
import getopt
import math
import os
import nltk
from nltk.corpus import stopwords
import pickle
import csv
import functools
from collections import Counter, defaultdict
from encode import encode
from enum import IntEnum

# Self-defined constants, functions and classes

# For Rocchio Coefficients
K = 14
ENG_STOPWORDS = set(stopwords.words('english'))

def filter_punctuations(s, keep_quo=False):
    """
    Takes in String s and returns the processed version of it
    Replaces certain punctuations with space, to be removed later on
    Removes others
    Set the 2nd argument to be True to keep quotation marks
    """
    space_wo_quo = '''!?;:\\.*+=_~<>[]{}(/)'''
    space_w_quo = '''!?;:\\.*+=_~<>[]{}(/")''' # same as space_wo_quo but now has ' " ' inside
    remove = """-'""" # e.g. apostrophe. Add in more if needed

    # Note: replacing any character with a space will incur a " " term (a space)
    # We remove this space in the generate_list_of_words function

    if keep_quo:
        for character in s:
            if character in remove:
                s = s.replace(character, "") # eg Arnold's Fried Chicken -> Arnolds Fried Chicken (more relevant) VS Arnold s Fried Chicken
            elif character in space_wo_quo:
                s = s.replace(character, " ") # will keep double inverted commas
    else:
        for character in s:
            if character in remove:
                s = s.replace(character, "")
            elif character in space_w_quo:
                s = s.replace(character, " ") # will remove double inverted commas
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
    # Otherwise, same term & doc_id, differentiate by field: title -> court -> date_posted -> content
    elif arr1[1][1] != arr2[1][1]:
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
        self.dictionary = {}  # content, title, court, date_posted
        self.docid_term_mappings = {} # (doc_id:{top K most common terms:their count} for that doc_id) mappings
        self.in_dir = in_dir
        self.d_file = d_file
        self.p_file = p_file

        global ENG_STOPWORDS
        ENG_STOPWORDS = self.process_words(ENG_STOPWORDS)

    def build(self):
        """
        Builds the Vector Space Model (VSM), which includes a dictionary of PostingLists for each term
        A dictionary of document lengths and a list of document ids are also made
        These are accessed via .dictionary, .doc_lengths, .doc_ids respectively
        Punctuation handling, tokenisation, case-folding, stemming are applied to generate terms
        """
        # Step 1: Obtain a list of all individual documents from the csv input file
        # Each complete document is represented by a dictionary with keys: 'doc_id', 'title', 'content', 'date_posted', 'court'
        # and keys for 4 positional_indexes: 'content_positional_indexes', 'title_positional_indexes', 'court_positional_indexes', 'date_posted_positional_indexes'
        set_of_documents = self.get_documents()

        # Step 2: Obtain all possible Postings and sort them by term, then doc_id, then by the required zones/fields

        # Save flattened [term, (doc_ID, Field, positional_index)] entries (of the zone/field positional indexes' PostingLists) in tokens_list for sorting
        tokens_list = []

        for single_document in set_of_documents:
            # Every document in here is unique and not repeated
            doc_id = single_document['doc_id']
            # Obtain all [term, (doc_ID, Field, positional_index)] entries
            # Add these into tokens_list for sorting
            tokens_list.extend(self.generate_token_list(doc_id, Field.CONTENT, single_document['content_positional_indexes']))
            tokens_list.extend(self.generate_token_list(doc_id, Field.TITLE, single_document['title_positional_indexes']))
            tokens_list.extend(self.generate_token_list(doc_id, Field.COURT, single_document['court_positional_indexes']))
            tokens_list.extend(self.generate_token_list(doc_id, Field.DATE_POSTED, single_document['date_posted_positional_indexes']))
            # For Rocchio Algo/Query Optimisation later on
            # Note that we can still access
            self.docid_term_mappings[doc_id] = single_document['top_K']

        # Sort the list of [term, (doc_ID, Field, positional_index)] entries
        tokens_list.sort(key=functools.cmp_to_key(comparator))

        # Step 3 (done as part of Step 2): Initialise a mapping of all available doc_ids to their most common terms
        # This is to facilitate query optimisation/refinement later on during search

        # Step 4: # Create a PostingList for every single term and fill it up with entries regardless of which zone/field
        # A dictionary will contain all the PostingLists, accessible by the key == term
        print("Generating posting lists")
        for i in range(len(tokens_list)):

            # Extract required information
            curr = tokens_list[i]
            term = curr[0]
            curr_tuple = curr[1] # (doc_id, Field, positional_index)

            # Create a new PostingList if this is a new term
            if i == 0 or term != tokens_list[i-1][0]:
                self.dictionary[term] = PostingList()

            # Insert into appropriate PostingList
            # If same term and docID, do not increment PostingList.size
            # Therefore, different zones/fields with same doc_id will still count as 1 doc_id in total
            if i > 0 and term == tokens_list[i-1][0] and curr_tuple[0] == tokens_list[i-1][1][0]:
                self.dictionary[term].insert(curr_tuple[0], curr_tuple[1], curr_tuple[2], False)
            else:
                self.dictionary[term].insert(curr_tuple[0], curr_tuple[1], curr_tuple[2])

        # Step 5: Calculate doc_lengths for normalization
        print("Calculating document vector length")
        self.calculate_doc_length()

    def get_documents(self):
        """
        Returns a list of complete documents which have positional indexes for content, title, court, and date_posted
        Each complete document is represented by a dictionary with keys: 'doc_id', 'title', 'content', 'date_posted', 'court', 'top_K'
        and keys for 4 positional_indexes: 'content_positional_indexes', 'title_positional_indexes', 'court_positional_indexes', 'date_posted_positional_indexes'
        """
        # Result container for collating all possible dictionary file terms
        set_of_documents = []
        documents = self.process_file()
        print("Done processing file")

        # Then we handle the fields: content, title and date_posted and court (to generate position index)
        count = 0
        for document in documents:
            document['content_positional_indexes'] = self.generate_positional_indexes(document['content'])  # Part 1: Content
            document['title_positional_indexes'] = self.generate_positional_indexes(document['title'])  # Part 2: Title
            document['court_positional_indexes'] = self.generate_positional_indexes(document['court'])  # Part 3: Court
            document['date_posted_positional_indexes'] = self.generate_positional_indexes(document['date_posted'].split()[0])  # Part 4: Date_posted

            # To obtain the top K terms for the current document
            accumulate_counts = {}
            self.include_count_contribution_from_pos_ind(accumulate_counts, document['content_positional_indexes'])
            self.include_count_contribution_from_pos_ind(accumulate_counts, document['title_positional_indexes'])
            self.include_count_contribution_from_pos_ind(accumulate_counts, document['court_positional_indexes'])
            self.include_count_contribution_from_pos_ind(accumulate_counts, document['date_posted_positional_indexes'])
            document['top_K'] = Counter(accumulate_counts).most_common(K)
            for i in range(K):
                # i must always be smaller than actual_size by 1
                # accumulate_counts has a possibility of going below K
                # to avoid null pointer exception, we use < len(accumulate_counts)
                if (i < len(accumulate_counts)):
                    document['top_K'][i] = document['top_K'][i][0]
                else:
                    break;

            # Now, document['top_K'] will be a list of the top K terms for the document
            set_of_documents.append(document)

            print(count," Generated positional indexes")
            count += 1

        print("Done getting documents")
        return set_of_documents

    def include_count_contribution_from_pos_ind(self, result_counts, pos_ind):
        """
        Finds each term's counts in the pos_ind dictionary and reflects this count contribution in the result_counts dictionary
        """
        for term in pos_ind:
            if term not in ENG_STOPWORDS:
                counts = len(pos_ind[term])
                if term in result_counts:
                    result_counts[term] += counts
                else:
                    result_counts[term] = counts

    def process_file(self):
        with open(self.in_dir, encoding='utf-8') as f:
            """
            Returns a list of documents (aka legal cases) by splitting the csv file into a list of documents
            Each document is represented by a dictionary with keys: 'doc_id', 'title', 'content', 'date_posted', 'court'
            Note: This function merely classifies the appropriate fields/zones, and DOES NOT filter punctuation or casefolds to lowercase
            Note: The documents created are intermediate documents, which are meant to have other values build in them later on in the get_documents function
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

            csv_reader = csv.reader(f, delimiter=',') # Read the file in and split by the characters '",'
            documents = []

            index = 0
            for row in csv_reader:
                if index != 0:
                    # this is a fresh new legal case/document
                    document = {}
                    # Renaming columns here so we cant use csv.DictReader
                    document['doc_id'] = int(row[0].strip(''))
                    document['title'] = row[1].strip('')
                    document['content'] = row[2].strip('')
                    document['date_posted'] = row[3].strip('')
                    document['court'] = row[4].strip('')
                    documents.append(document)
                index += 1

            return documents

    def generate_positional_indexes(self, paragraph):
        """
        Generates a positional index (positions stored via gap encoding) for a field/zone (from its paragraph/string)
        by using the function generate_list_of_words() and generate_positional_indexes_from_list()
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
        words_array = [nltk.word_tokenize(s) for s in sentences]
        words = [filter_punctuations(w) for arr in words_array for w in arr] # ensure consistency with search.py
        words = [w for w in words if w != " "]
        processed_words = self.process_words(words)
        return processed_words

    def process_words(self, words):
        """
        Stems the already lowercase version of the word given and lowercases
        Takes in the list of Strings and returns their stemmed version in a list
        """
        stemmer = nltk.stem.porter.PorterStemmer()
        return [stemmer.stem(w.lower()) for w in words]

    def generate_positional_indexes_from_list(self, words, start_index):
        """
        Generate the positional indexes for the phrasal queries from a list of words
        Positions are stored via gap encoding, so only the first position is the absolute position
        Subsequent positions are obtained via addition from the previous
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

    def generate_token_list(self, doc_id, field_type, positional_indexes):
        """
        Generates entries from a list of positional index
        """
        tokens_list = []
        for term, positional_index in positional_indexes.items():
            tokens_list.append([term, (doc_id, field_type, positional_index)])  # [term, (doc_ID, Field, positional_indexes)]

        return tokens_list

    def calculate_doc_length(self):
        """
        Sets and stores the length of each document for use during normalization
        """
        # Iterate through every term of the dictionary, getting its PostingList
        # We iterate every posting in the PostingList and calculate its contribution to its document's vector length
        # This contribution is accumualted, and then square-rooted to find the vector's length, used for normalisation later on
        self.doc_lengths = {}
        for _, posting_list in self.dictionary.items():

            # This is done at term-level (aka for every term, aka using each PostingList)

            # This stores doc_id:total_tf mappings
            tf_overall = {}

            # Accumulate the total_tf values
            for posting in posting_list.postings:
                tf_contribution = len(posting.positions) # tf contribution for current term from current zone/field
                if (posting.doc_id) not in tf_overall:
                    tf_overall[posting.doc_id] = tf_contribution
                else:
                    tf_overall[posting.doc_id] += tf_contribution

            # Since each term has a non-zero tf contribution to give a non-zero length contribution (due to lnc document weighting scheme)
            # to the length of the document vector if the term appears in the document vector,
            # we calculate this length contribution to the document length
            for id, tf in tf_overall.items():
                posting_weight = 1 + math.log(tf, 10) # lnc for documents, tf = previously accumulated tf value is guarenteed > 0
                if id not in self.doc_lengths:
                    self.doc_lengths[id] = posting_weight * posting_weight
                else:
                    self.doc_lengths[id] += (posting_weight * posting_weight)

        # Square-root to find the vector length
        for doc_id, total_weight in self.doc_lengths.items():
            self.doc_lengths[doc_id] = math.sqrt(total_weight)

    def write(self):
        """
        Writes PostingList objects into postings file and all terms into dictionary file
        doc_lengths and docid_term_mappings are also written into dictionary file
        """

        d = {}  # to contain mappings of term to file cursor value
        with open(self.p_file, "wb") as f:
            for word, posting_list in self.dictionary.items():
                cursor = f.tell()
                d[word] = cursor # updating respective (term to file cursor value) mappings
                pickle.dump(posting_list, f, protocol=4)

        with open(self.d_file, "wb") as f:
            pickle.dump(d, f) # (term to file cursor value) mappings dictionary
            pickle.dump(self.doc_lengths, f) # document lengths regardless of zone/field types
            pickle.dump(self.docid_term_mappings, f) # (doc_id to K most common terms) mappings

class Field(IntEnum):
    """
    Represents the possible fields for a document given other than the doc_id
    """
    CONTENT = 1
    TITLE = 2
    COURT = 3
    DATE_POSTED = 4

class Posting:
    """
    Each Posting has a document id (doc_id), field type (field), positional index (positions), and a pointer for possible optimisation
    Note that term frequency can be obtained by len(positions)
    Each Posting represents a document for a particular term
    """
    def __init__(self, index, doc_id, field, positions):
        self.doc_id = doc_id
        self.field = field
        self.positions = positions

    def var_byte_encoding(self):
        self.positions = encode(self.positions)

    def generate_string_of_posting(self):
        return ' (' + str(self.doc_id) + ', ' + str(self.field) + ', ' + str(self.positions) + ') '

class PostingList:
    """
    Each PostingList is a collection of Postings for a particular term.
    A PostingList contains the number of unique documents it contains regardless of which zone/field (size) and a list of Postings (postings)
    """
    def __init__(self):
        self.postings = []
        self.unique_docids = 0

    def get_unique_docids(self):
        return self.unique_docids

    # Insert with var byte encoding
    def insert(self, doc_id, field, positions, new_doc_id=True):
        next_id = self.unique_docids
        new_posting = Posting(next_id, doc_id, field, positions)
        new_posting.var_byte_encoding()
        self.postings.append(new_posting)
        if new_doc_id:
            self.unique_docids += 1

    def insert_without_encoding(self, doc_id, field, positions):
        next_id = self.unique_docids
        self.postings.append(Posting(next_id, doc_id, field, positions))
        self.unique_docids += 1

    def insert_posting(self, posting):
        self.postings.append(posting)

    def get(self, index):
        return self.postings[index]

    def generate_string_of_postinglist(self):
        s = 'size ' + str(self.unique_docids) + '  '
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
