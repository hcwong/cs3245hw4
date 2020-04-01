#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import pickle
import math
import heapq
import functools
from collections import Counter
from index import Posting, PostingList

# Initialise Global variables

D = {} # to store all (term to posting file cursor value) mappings
POSTINGS_FILE_POINTER = None # reference for postings file
DOC_LENGTHS = None # to store all document lengths
ALL_DOC_IDS = None # to store all doc_ids
K = 10 # Number of search results to display to the user
AND_KEYWORD = "AND"

def comparator(tup1, tup2):
    """
    Sorts the 2 tuples by score first, then doc_id in ascending order
    """
    if tup1[0] > tup2[0]:
        return 1
    elif tup2[0] > tup1[0]:
        return -1
    else:
        return tup2[1] - tup1[1]

# Parsing

def process(line):
    """
    Filters out some punctuations and then case-folds to lowercase
    Takes in a String line and returns the result String
    """
    return filter_punctuations(line).lower()

def filter_punctuations(s):
    """
    Replaces certain punctuations from Strings with space, to be removed later on
    Takes in a String s and returns resultant String
    """
    punctuations = '''!?-;:"\\,./#$%^&<>[]{}*_~()'''
    for character in s:
        if character in punctuations:
            s = s.replace(character, " ")
    return s

def tokenize_and_process_query(query):
    """
    Takes in a case-folded String previously processed for punctuations, tokenises it, and returns a list of stemmed terms
    """
    stemmer = nltk.stem.porter.PorterStemmer()
    return [stemmer.stem(term) for term in nltk.word_tokenize(query)]

# Ranking

def cosine_score(tokens_arr):
    """
    Takes in an array of terms, and returns a list of the top K scoring documents based on cosine similarity scores with respect to the query terms
    """
    # For every query term, cosine similarity contributions are made only for documents containing the query term
    # To optimise, we only do calculation for these documents and doing so pointwise.
    # Here, we obtain score contributions term-wise and accumulate them before moving onto the next term,
    # rather than wait to do so only after constructing the query vector which incurs overhead.
    # We normalise at the end to optimise speed.

    scores = {} # to store all cosine similarity scores for each query term
    term_frequencies = Counter(tokens_arr) # the query's count vector for every of its terms, to obtain data for pointwise multiplication

    for term in tokens_arr:
        # 1. Obtain the first vector, representing all the document(s) containing the term
        # We will calculate its weight in Step 3
        # This is nicely reflected in the term's PostingList
        # Only documents with Postings of this term will have non-zero score contributions
        posting_list = find_term(term)
        # Invalid query terms have no Postings and hence no score contributions;
        # in this case we advance to the next query term saving unnecessary operations
        if (posting_list == []):
            continue

        # 2. Obtain the second vector's (query vector's) value for pointwise multiplication
        # Calculate the weight entry of the term in the query, of the term-document matrix
        query_term_weight = get_query_weight(len(posting_list), term_frequencies[term])

        # 3. Perform pointwise multiplication for the 2 vectors
        # The result represents the cosine similarity score contribution from the current term before normalisation
        # Accumulate all of these contributions to obtain the final score before normalising
        for posting in posting_list:
            # Obtain pre-computed weight of term for each document and perform calculation
            doc_term_weight = posting.weight # guaranteed no error in log calculation as tf >= 1
            if posting.doc_id not in scores:
                scores[posting.doc_id] = (doc_term_weight * query_term_weight)
            else:
                scores[posting.doc_id] += (doc_term_weight * query_term_weight)

    # 4. Perform normalisation to consider the length of the document vector
    # We save on dividing by the query vector length as it is constant for all documents
    # and therefore does not affect comparison of scores
    results = []
    for doc_id, total_weight in scores.items():
        ranking_score = total_weight / DOC_LENGTHS[doc_id]
        results.append((ranking_score, doc_id))

    # 5. Sort the documents by score then doc_id (if same score) and return the top K highest scoring ones
    heapq.heapify(results)
    if len(results) >= K:
        return [x[1] for x in heapq.nlargest(K, results, key=functools.cmp_to_key(comparator))]
    else:
        return [x[1] for x in heapq.nlargest(len(results), results, key=functools.cmp_to_key(comparator))]

def get_query_weight(df, tf):
    """
    Calculates the tf-idf weight for a term in the query vector
    Takes in document frequency df, term frequency tf, and returns the resulting tf-idf weight
    We treat the query as a document itself, having its own term count vector
    We use ltc in the calculation for queries, as opposed to lnc for documents
    This requires document frequency df, term frequency tf, total number of documents N
    """
    N = len(ALL_DOC_IDS)
    # df, tf and N are all guranteed to be at least 1, so no error is thrown here
    return (1 + math.log(tf, 10)) * math.log(N/df, 10)

def find_term(term):
    """
    Takes in a term, then finds and returns the list representation of the PostingList of the given term
    or an empty list if no such term exists in index
    """
    term = term.strip()
    if term not in D:
        return []
    POSTINGS_FILE_POINTER.seek(D[term])
    return pickle.load(POSTINGS_FILE_POINTER).postings

# Takes in a phrasal query in the form of an array of terms and returns the doc ids which have the phrase
# Note: Only use this for boolean retrieval, not free text mode
def perform_phrase_query(phrase):
    # Defensive programming, if phrase is empty, return false
    if not phrase:
        return False
    phrase_posting_list = find_term(phrase[0])
    for term in phrase[1:]:
        current_term_postings = find_term(term)
        # Order of arguments matter
        phrase_posting_list = merge_posting_lists(phrase_posting_list, current_term_postings, True)

    return phrase_posting_list

# Returns merged positions for phrasal query
# positions2 comes from the following term and positions1 from
# the preceeding term
def merge_positions(positions1, positions2):
    merged_positions = []
    L1 = len(positions1)
    L2 = len(positions2)
    curr1, curr2 = 0, 0
    while curr1 < L1 and curr2 < L2:
        if positions1[curr1] + 1 == positions2[curr2]:
            # Only merge the position of curr2 because
            # We only need the position of the preceeding term
            merged_positions.append(positions2[curr2])
            curr1 += 1
            curr2 += 1
        elif positions1[curr1] + 1 > positions2[curr2]:
            curr2 += 1
        else:
            curr1 += 1
    return merged_positions

# Performs merging of two postings
def merge_posting_lists(list1, list2, should_perform_merge_positions = False):
    """
    Merges list1 and list2 for the AND boolean operator
    """
    merged_list = PostingList()
    L1 = len(list1)
    L2 = len(list2)
    curr1, curr2 = 0, 0

    while curr1 < L1 and curr2 < L2:
        posting1 = list1[curr1]
        posting2 = list2[curr2]
        # If both postings have the same doc id, add it to the merged list.
        if posting1.doc_id == posting2.doc_id:
            curr1 += 1
            curr2 += 1
            if should_perform_merge_positions:
                merged_positions = merge_positions(posting1.positions, posting2.positions)
                # Only add the doc_id if the positions are not empty
                if len(merged_positions > 0):
                    merged_list.insert(posting1.doc_id, merged_positions)
            else:
                merged_list.insert_posting(posting1)
        else:
            # Else if there is a opportunity to jump and the jump is less than the doc_id of the other list
            # then jump, which increments the index by the square root of the length of the list
            if posting1.pointer != None and posting1.pointer.doc_id < posting2.doc_id:
                curr1 = posting1.pointer.index
            elif posting2.pointer != None and posting2.pointer.doc_id < posting1.doc_id:
                curr2 = posting2.pointer.index
            # If we cannot jump, then we are left with the only option of incrementing the indexes one by one
            else:
                if posting1.doc_id < posting2.doc_id:
                    curr1 += 1
                else:
                    curr2 += 1
    return merged_list

def parse_query(query):
    phrasal_regex_pattern = '.*\"(.*)\".*'
    if AND_KEYWORD in query or re.search(phrasal_regex_pattern, query):
        return parse_boolean_query

    else:
        return parse_free_text_query(query)

def parse_boolean_query(query):
    parse_by_arr = query.split(AND_KEYWORD)
    stemmer = nltk.stem.porter.Stemmer()
    parse_by_arr = [stemmer.stem(word.lower()) for word in parse_by_arr]
    if not parse_by_arr:
        return []

    first_term = parse_by_arr[0]
    res_posting_list = None
    if " " in first_term:
        res_posting_list = perform_phrase_query(first_term)
    else:
        res_posting_list = find_term(first_term)

    for term in parse_by_arr[1:]:
        term_posting_list = None
        if " " in term:
            term_posting_list = perform_phrase_query(term)
        else:
            term_posting_list = find_term(term)
        res_posting_list = merge_posting_lists(res_posting_list, term_posting_list)
    return res_posting_list

def parse_free_text_query(query):
    tokens = tokenize_and_process_query(process(query))
    res = cosine_score(tokens)
    return res


# Below are the code provided in the original Homework search.py file, with edits to run_search to use our implementation

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

def run_search(dict_file, postings_file, queries_file, results_file):
    """
    Perform query searches from queries file using the given dictionary file and postings file, writing results to results file
    """
    global D
    global POSTINGS_FILE_POINTER
    global DOC_LENGTHS
    global ALL_DOC_IDS

    # 1. Reading data from files into memory: File Pointer Mappings, Document Lengths, Document IDs
    dict_file_fd = open(dict_file, "rb")
    POSTINGS_FILE_POINTER = open(postings_file, "rb")
    D = pickle.load(dict_file_fd) # dictionary with term:file cursor value entries
    DOC_LENGTHS = pickle.load(dict_file_fd) # dictionary with doc_id:length entries
    ALL_DOC_IDS = pickle.load(POSTINGS_FILE_POINTER) # data for optimisation, if needed
    # PostingLists for each term are accessed separately using file cursor values given in D
    # because they are significantly large and unsuitable for all of them to be used in-memory

    # 2. Process Queries
    with open(queries_file, "r") as q_file:
        with open(results_file, "w") as r_file:
            for line in q_file:
                # Each line is a query search
                # Parse the initial query: filter punctuations, case-folding
                line = line.rstrip("\n")
                res = [] # contains all possible Postings for the result of a single line query
                try:
                    tokens = tokenize_and_process_query(line)
                    # Obtain the K top scoring doc_ids based on cosine similarity score
                    # Note: Postings are in the top 10 scores unless truncated or fully exchausted
                    res = cosine_score(tokens)
                except Exception as e:
                    print(repr(e), line)
                    res = [] # invalid search queries

                # 3. Write the K most relevant documents for the current query to storage
                # in descending score order, then ascending doc_id
                r_file.write(" ".join([str(r) for r in res]) + "\n")

    # 4. Cleaning up: close files
    dict_file_fd.close()
    POSTINGS_FILE_POINTER.close()

dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file  = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
