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
from index import Posting, PostingList, Field
from encode import check_and_decode
from nltk.corpus import wordnet
from nltk.corpus import stopwords

# Initialise Global variables

D = {} # to store all (term to posting file cursor value) mappings
POSTINGS_FILE_POINTER = None # reference for postings file
DOC_LENGTHS = None # to store all document lengths
ALL_DOC_IDS = None # to store all doc_ids
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
def filter_punctuations(s):
    """
    Replaces certain punctuations from Strings with space, to be removed later on
    Takes in String s and returns the processed version of it
    """
    punctuations = '''!?-;:\\,./#$%^&<>[]{}*_~()'''
    filtered_term = ""
    for character in s:
        if character in punctuations:
            filtered_term += " "
        else:
            filtered_term += character
    return filtered_term

def process(arr):
    """
    Filters out some punctuations and then case-folds to lowercase
    Takes in a String array and returns the result array
    """
    return [filter_punctuations(term) for term in arr]

def stem_word(term):
    stemmer = nltk.stem.porter.PorterStemmer()
    return stemmer.stem(term)

def stem_query(arr):
    """
    Takes in a case-folded array of terms previously processed for punctuations, tokenises it, and returns a list of stemmed terms
    """
    return [stem_word(term) for term in arr]

# Ranking

def boost_score_based_on_field(field, score):
    # TODO: Decide on an appropriate boost value
    # court_boost = 3
    # title_boost = 10
    # if field == Field.TITLE:
        # return score * title_boost
    # elif field == Field.COURT:
        # return score * court_boost
    # else:
        # return score
    return score

def cosine_score(tokens_arr):
    """
    Takes in an array of terms, and returns a list of the top scoring documents based on cosine similarity scores with respect to the query terms
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
        posting_list = None
        if " " in term:
            posting_list_object = perform_phrase_query(term)
            if posting_list_object is not None:
                posting_list = posting_list_object.postings
        else:
            posting_list = find_term(term)
        # Invalid query terms have no Postings and hence no score contributions;
        # in this case we advance to the next query term saving unnecessary operations
        if posting_list is None:
            continue

        # 2. Obtain the second vector's (query vector's) value for pointwise multiplication
        # Calculate the weight entry of the term in the query, of the term-document matrix
        query_term_weight = get_query_weight(posting_list.size, term_frequencies[term])

        # 3. Perform pointwise multiplication for the 2 vectors
        # The result represents the cosine similarity score contribution from the current term before normalisation
        # Accumulate all of these contributions to obtain the final score before normalising
        # Accumulate all of these contributions to obtain the final score before normalising
        for posting in posting_list.postings:
            # Obtain pre-computed weight of term for each document and perform calculation
            doc_term_weight = 1 + math.log(len(posting.positions), 10) # guaranteed no error in log calculation as tf >= 1
            if posting.doc_id not in scores:
                scores[posting.doc_id] = (boost_score_based_on_field(posting.field, doc_term_weight) * query_term_weight)
            else:
                scores[posting.doc_id] += (boost_score_based_on_field(posting.field, doc_term_weight) * query_term_weight)

    doc_ids_in_tokens_arr = find_by_document_id(tokens_arr)

    # 4. Perform normalisation to consider the length of the document vector
    # We save on dividing by the query vector length as it is constant for all documents
    # and therefore does not affect comparison of scores
    results = []
    for doc_id, total_weight in scores.items():
        ranking_score = total_weight / DOC_LENGTHS[doc_id]
        # Now we check if any of the query terms matches 
        # TODO: Improve the doc_id criterion below
        if doc_id in doc_ids_in_tokens_arr:
            ranking_score += 100 # RANDOM VALUE
        results.append((ranking_score, doc_id))


    # Sort the results in descending order of score
    results.sort(key=functools.cmp_to_key(comparator))

    return results

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
    # NOTE: LOWERCASING IS ONLY DONE HERE.
    term = term.strip().lower()
    term = stem_word(term)
    if term not in D:
        return None
    POSTINGS_FILE_POINTER.seek(D[term])
    return pickle.load(POSTINGS_FILE_POINTER)

def find_by_document_id(terms):
    """
    Checks if any of the query terms are document ids, if so return the document id
    To be used after the normal boolean/free text parsing
    """
    document_ids = []
    for term in terms:
        if all(map(str.isdigit, term)):
            if int(term) in ALL_DOC_IDS:
                document_ids.append(int(term))
    return document_ids

# Takes in a phrasal query in the form of an array of terms and returns the doc ids which have the phrase
# Note: Only use this for boolean retrieval, not free text mode
def perform_phrase_query(phrase_query):
    # Defensive programming, if phrase is empty, return false
    if not phrase_query:
        return False
    phrases = phrase_query.split(" ")
    phrase_posting_list = find_term(phrases[0])
    if phrase_posting_list == None:
        return None

    for term in phrases[1:]:
        current_term_postings = find_term(term)
        if current_term_postings == None:
            return None
        # Order of arguments matter
        phrase_posting_list = merge_posting_lists(phrase_posting_list, current_term_postings, True)

    return phrase_posting_list

# Returns merged positions for phrasal query
# positions2 comes from the following term and positions1 from
# the preceeding term
def merge_positions(positions1, positions2, doc_id):
    merged_positions = []
    L1 = len(positions1)
    L2 = len(positions2)
    index1, index2 = 0, 0
    offset1, offset2 = 0, 0
    # This is for our gap encoding
    last_position_of_merged_list = 0
    # Do this because we have byte encoding
    calculate_actual_pos_from_offset = lambda curr_value, offset: curr_value + offset
    while index1 < L1 and index2 < L2:
        proper_position2 = calculate_actual_pos_from_offset(positions2[index2], offset2)
        if calculate_actual_pos_from_offset(positions1[index1], offset1) + 1 == proper_position2:
            # Only merge the position of index2 because
            # We only need the position of the preceeding term
            # Need to do some math now because of our gap encoding, sadly
            position_to_append = proper_position2 - last_position_of_merged_list
            last_position_of_merged_list = proper_position2
            merged_positions.append(position_to_append)

            # Update the offsets of the original two positing lists
            offset1 += positions1[index1]
            offset2 += positions2[index2]
            index1 += 1
            index2 += 1
        elif calculate_actual_pos_from_offset(positions1[index1], offset1) + 1 > proper_position2:
            offset2 += positions2[index2]
            index2 += 1
        else:
            offset1 += positions1[index1]
            index1 += 1
    return merged_positions

# Performs merging of two posting lists
# Note: Should perform merge positions is only used for phrasal queries
# Term frequency does not matter for normal boolean queries
def merge_posting_lists(list1, list2, should_perform_merge_positions = False):
    """
    Merges list1 and list2 for the AND boolean operator
    """
    merged_list = PostingList()
    L1 = list1.size
    L2 = list2.size
    curr1, curr2 = 0, 0

    while curr1 < L1 and curr2 < L2:
        posting1 = list1.postings[curr1]
        posting2 = list2.postings[curr2]
        # If both postings have the same doc id, add it to the merged list.
        if posting1.doc_id == posting2.doc_id:
            # Order of fields is title -> court-> content
            # Now we have to merge by the postings of the different fields
            # Case 1: Both doc_id and field are the same
            if posting1.field == posting2.field:
                if should_perform_merge_positions:
                    merged_positions = merge_positions(check_and_decode(posting1.positions), check_and_decode(posting2.positions), posting1.doc_id)
                    # Only add the doc_id if the positions are not empty
                    if len(merged_positions) > 0:
                        merged_list.insert_without_encoding(posting1.doc_id, posting1.field, merged_positions)
                else:
                    merged_list.insert_posting(posting1)
                curr1 += 1
                curr2 += 1
            # Case 2: posting1's field smaller than posting2's field
            elif posting1.field < posting2.field:
                # TODO: To prove but I think this hunch is correct
                # There should not be a case where posting2 has the same field but has merged it in previously.
                # This insert should never be a duplicate
                merged_list.insert_posting(posting1)
                curr1 += 1
            # Case 3: Converse of case 2
            else:
                merged_list.insert_posting(posting2)
                curr2 += 1
        else:
            # Else if there is a opportunity to jump and the jump is less than the doc_id of the other list
            # then jump, which increments the index by the square root of the length of the list
            # if posting1.pointer != None and posting1.pointer.doc_id < posting2.doc_id:
                # curr1 = posting1.pointer.index
            # elif posting2.pointer != None and posting2.pointer.doc_id < posting1.doc_id:
                # curr2 = posting2.pointer.index
            # # If we cannot jump, then we are left with the only option of incrementing the indexes one by one
            # else:
            if posting1.doc_id < posting2.doc_id:
                curr1 += 1
            else:
                curr2 += 1
    return merged_list

def parse_query(query, relevant_docids):
    terms_array, is_boolean_query = split_query(query)
    if is_boolean_query:
        return parse_boolean_query(terms_array, relevant_docids)
    else:
        return parse_free_text_query(terms_array, relevant_docids)

def get_ranking_for_boolean_query(posting_list, relevant_docids):
    """
    The scoring for boolean queries is going to follow CSS Specificity style
    Title matches will be worth 20, court 10 and content 1 (numbers to be confirmed)
    The overall relevance of the documents would be the sum of all these scores
    Example: If the resultant posting list has two postings for doc_id xxx, with fields COURT and CONTENT
    Then the resultant score is 11
    """
    relevant_score = 100000
    title_score = 20
    court_score = 10
    content_score = 1

    def get_boolean_query_scores(field):
        if field == Field.TITLE:
            return title_score
        elif field == Field.COURT:
            return court_score
        else:
            return content_score

    scores = {}
    for posting in posting_list.postings:
        score = get_boolean_query_scores(posting.field)
        if posting.doc_id not in scores:
            scores[posting.doc_id] = score
        else:
            scores[posting.doc_id] += score

    # Add to the ones judged relevant by humans
    for relevant_docid in relevant_docids:
        if relevant_docid in scores:
            scores[relevant_docid] = relevant_score
        else:
            scores[relevant_docid] += relevant_score

    # Now we do the sorting
    sorted_results = sorted([(score, doc_id) for doc_id, score in scores.items()], key=functools.cmp_to_key(comparator))

    return sorted_results

def parse_boolean_query(terms, relevant_docids):
    """
    Takes in the array of terms from the query
    Returns the posting list of all the phrase
    """
    # First filter out all the AND keywords from the term array
    filtered_terms = [term for term in terms if term != AND_KEYWORD]

    filtered_terms = process(filtered_terms)
    # Get the posting list of the first word
    first_term = filtered_terms[0]
    res_posting_list = None
    if " " in first_term:
        res_posting_list = perform_phrase_query(first_term)
    else:
        res_posting_list = find_term(first_term)

    if res_posting_list is None:
        return []

    # Do merging for the posting lists of the rest of the terms
    for term in filtered_terms[1:]:
        term_posting_list = None
        if " " in term:
            term_posting_list = perform_phrase_query(term)
        else:
            term_posting_list = find_term(term)

        if term_posting_list is None:
            return []
        res_posting_list = merge_posting_lists(res_posting_list, term_posting_list)

    return get_ranking_for_boolean_query(res_posting_list, relevant_docids)

def parse_free_text_query(terms, relevant_docids):
    # TODO: See below (delete once done)
    #Expected to add query expansion, after process(query) is done
    #query = query_expansion(process(query))
    terms = process(terms)
    res = cosine_score(terms)
    return res

def split_query(query):
    """
    split_query extracts out the terms into phrases and terms
    Assumes that the query is well formed.
    """
    start_index = 0
    is_in_phrase = False
    is_boolean_query = False
    current_index = 0
    terms = []

    while current_index < len(query):
        current_char = query[current_index]
        if current_char == "\"":
            if is_in_phrase:
                is_in_phrase = False
                terms.append(query[start_index:current_index])
                start_index = current_index + 1 # +1 to ignore the space after this
            else:
                start_index = current_index + 1
                is_in_phrase = True
        elif current_char == " ":
            # Append the word if not parsing part of phrase
            if not is_in_phrase:
                terms.append(query[start_index:current_index])
                if (query[start_index:current_index] == AND_KEYWORD):
                    is_boolean_query = True
                start_index = current_index + 1
        current_index += 1

    # Add in the last term if it exists
    if start_index < current_index:
        terms.append(query[start_index:current_index])

    # Weed out empty strings
    return [term for term in terms if term], is_boolean_query

def query_expansion(query):
    #Split the query into words
    #Remove stop words
    #Find the synonyms of each word and append them to a set, since some of the synonyms might be repetitive
    #Add the set of synonyms to list of extended query words
    #Convert the extended query list to extende query string
    #Return the string

    query_words = query.split()
    stop_words = set(stopwords.words('english'))
    query_words = [word for word in query_words if not word in stop_words]
    expanded_query = []
    for word in query_words:
        expanded_query.append(word)
        syn_set = set()
        for s in wordnet.synsets(word):
            for l in s.lemmas():
                syn_set.add(l.name())
        expanded_query.extend(syn_set)

    new_query = ' '.join([str(word).lower() for word in expanded_query])

    return new_query
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
    ALL_DOC_IDS = pickle.load(dict_file_fd) # data for optimisation, if needed
    # PostingLists for each term are accessed separately using file cursor values given in D
    # because they are significantly large and unsuitable for all of them to be used in-memory

    # 2. Process Queries
    with open(queries_file, "r") as q_file:
        with open(results_file, "w") as r_file:
            # TODO: Wrap these clause with Try catch block
            lines = [line.rstrip("\n") for line in q_file.readlines()]
            query = lines[0]
            relevant_docids = [int(doc_id) for doc_id in lines[1:]]
            res = []
            res = parse_query(query, relevant_docids)
            r_file.write(" ".join([str(r[1]) for r in res]) + "\n")

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
