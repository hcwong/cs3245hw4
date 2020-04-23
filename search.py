#!/usr/bin/python3
# -*- coding: utf-8 -*-

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
EMPHASIS_ON_ORIG = 1.0 # initial query
EMPHASIS_ON_RELDOC = 0.75 # relevant marked documents
EMPHASIS_ORIG_MULTIPLIER_POSTPROCESSING = 1.1

def comparator(tup1, tup2):
    """
    Sorts the 2 tuples by score first, then doc_id in ascending order
    """
    if tup1[0] > tup2[0]:
        return -1
    elif tup2[0] > tup1[0]:
        return 1
    else:
        return tup2[1] - tup1[1]

# Parsing
def filter_punctuations(s):
    """
    Takes in String s and returns the processed version of it
    Replaces certain punctuations with space, to be removed later on
    Removes others
    Note: We will never encounter double inverted commas here, as they are already
    removed in identifying phrases for phrasal queries
    """
    space = '''!?;:\\.*+=_~<>[]{}(/)'''
    remove = """'-""" # e.g. apostrophe. Add in more if needed

    # Note: replacing any character with a space will incur a " " term (a space)
    # We remove this space later on in the process function

    for character in s:
        if character in remove:
            s = s.replace(character, "")
        elif character in space:
            s = s.replace(character, " ") # will remove double inverted commas
    return s

def process(arr):
    """
    Filters out some punctuations and then case-folds to lowercase
    Takes in a String array and returns the result array
    """
    return [filter_punctuations(term) for term in arr if term != " "]

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
    court_boost = 4
    title_boost = 10
    if field == Field.TITLE:
        return score * title_boost
    elif field == Field.COURT:
        return score * court_boost
    else:
        # no boost to score
        return score
    return score

def cosine_score(tokens_arr, relevant_docids):
    """
    Takes in an array of terms, and returns a list of the top scoring documents based on cosine similarity scores with respect to the query terms
    """

    # We first obtain query vector value for specific term
    # Then, if needed, we perform Rocchio Algorithm to finalise the query vector based on relevance assessments
    # Once done, we calculate each term's score contribution to every one of its documents' overall score as with the standard VSM
    # This will reflect the effects of the Rocchio Algorithm (if performed prior)

    # Rocchio Algorithm:
    # 1. Take in original query vector value for this term
    # 2. Take in all relevant_docids vector values for this term, accumulate them
    # 3. Average/normalise the accumulated value to account for within-document distribution
    # 3. Use this as the new query vector's value for this term
    # note: This approach is done term-wise, and we iterate through the term's posting list
    # to be able to process all score contributions from each document that contains the term

    # We normalise at the end to optimise speed.

    # Step 1: Preparation
    scores = {} # to store all cosine similarity scores for each query term
    term_frequencies = Counter(tokens_arr) # the query's count vector for every of its terms, to obtain data for pointwise multiplication
    # Set of all relevant documents' top K (already processed) terms, and query's processed terms

    # All terms inside are ALREADY PROCESSED aka filtered for punctuations, casefolded to lowercase, stemmed
    union_of_relevant_doc_top_terms = obtain_all_cos_score_terms(relevant_docids, tokens_arr)


    # Step 2: Obtain PostingList of interest
    is_entirely_phrasal = True # if False, should perform Rocchio for Query Refinement
    for term in tokens_arr:
        # Document IDs and zone/field types are nicely reflected in the term's PostingList
        # Only documents with Postings (regardless of zone/field type) of this term will have non-zero score contributions
        posting_list = None

        # At this point, phrasal queries will have terms which are phrases (has a space within);
        # otherwise, it is a freetext query -> Perform Rocchio Algorithm Query Refinement
        # note that in a mixture of both freetext and phrasal queries, we will perform Query Reginement
        query_type = "YET DECIDED"
        if " " in term:
            query_type = "PHRASAL"
            posting_list = perform_phrase_query(term) # do merging of PostingLists

        else:
            query_type = "FREETEXT"
            posting_list = find_term(term)
            is_entirely_phrasal = False # should perform Rocchio

        if posting_list is None:
            # Invalid query term: Move on to next
            continue


        # Step 3: Obtain the query vector's value for pointwise multiplication (Perform Relevance Feedback, if needed)
        query_term_weight = get_query_weight(posting_list.unique_docids, term_frequencies[term]) # before/without Rocchio

        # Query Refinement: Rocchio Algorithm (Part 1: common terms with query)
        # Want to use given relevant documents to get entry of the term in the refined query vector
        if (query_type == "FREETEXT") and (len(relevant_docids) != 0):
            # We are doing query refinement for this current term; no need to do again later: remove it first!
            # current term is not processed -> Need to process first to compare
            remove_term_processed_from_set(term, union_of_relevant_doc_top_terms)
            # calculate the centroid value for tf for calculating refined query value for this term
            # Note: documents can have a 0 contribution to the centroid value for a particular term if they don't contain it
            relevant_centroid_value = calculate_relevant_centroid_weight(relevant_docids, posting_list)
            # Use the relevant 'centroid' to calculate refined query entry
            if (relevant_centroid_value > 0):
                # most of the time, it should arrive at this branch
                query_term_weight = (EMPHASIS_ON_ORIG * query_term_weight) + (EMPHASIS_ON_RELDOC * relevant_centroid_value)
                # Otherwise, we don't change query_term_weight as it is better off without, or error in Rocchio Algo value

        # Step 4: Perform scoring by pointwise multiplication for the 2 vectors
        # Accumulate all score contribution from the current term before normalisation (done later) for lnc.ltc scheme
        # Boost accordingly to fields/zones
        for posting in posting_list.postings:
            doc_term_weight = 1 + math.log(len(posting.positions), 10) # guaranteed no error in lnc calculation as tf >= 1
            if posting.doc_id not in scores:
                scores[posting.doc_id] = (boost_score_based_on_field(posting.field, doc_term_weight) * query_term_weight)
            else:
                scores[posting.doc_id] += (boost_score_based_on_field(posting.field, doc_term_weight) * query_term_weight)

    # Step 5 (Optional): Rocchio Part 2 (if needed; for terms in overall top_K yet to be considered)
    # We begin with an initial query vector value of 0. And then we add the averaged lawyer-marked-relevant documents' 'centroid' value
    # Note the terms here are already processed; we need to use find_already_processed_term(term) function
    if (is_entirely_phrasal == False) and len(relevant_docids) != 0:
        # for terms that have not been covered, but need to be considered by Rocchio
        while (len(union_of_relevant_doc_top_terms) > 0):

            # Find PostingLists of terms until no more; removed from the set by .pop()
            next_term = union_of_relevant_doc_top_terms.pop()
            # Find posting list for the term
            posting_list = find_already_processed_term(next_term)
            if posting_list is None:
                continue # skip if invalid term

            # Calculate refined query value for multiplication
            # Initialised at 0 since ltc scheme gives 0 for query not containing current term
            # this is entirely made from contributions of the relevant documents
            final_query_value = (EMPHASIS_ON_RELDOC) * calculate_relevant_centroid_weight(relevant_docids, posting_list)

            # as before
            for posting in posting_list.postings:
                # Obtain weight of term for each document and perform calculation
                doc_term_weight = 1 + math.log(len(posting.positions), 10) # guaranteed no error in log calculation as tf >= 1
                if posting.doc_id not in scores:
                    scores[posting.doc_id] = (boost_score_based_on_field(posting.field, doc_term_weight) * final_query_value)
                else:
                    scores[posting.doc_id] += (boost_score_based_on_field(posting.field, doc_term_weight) * final_query_value)


    # Step 6: Perform normalisation to consider the length of the document vector
    # We save on dividing by the query vector length which is constant and does not affect score comparison
    doc_ids_in_tokens_arr = find_by_document_id(tokens_arr) # for manual post-processing to emphasis more on documents with original query terms further
    results = []
    for doc_id, total_weight in scores.items():
        ranking_score = total_weight/DOC_LENGTHS[doc_id]
        # Now we check if any of the query terms matches
        # TODO: Improve the doc_id criterion below
        # Since the user searches for terms which he/she tends to want, we place higher emphasis on these
        if doc_id in doc_ids_in_tokens_arr:
            ranking_score *= EMPHASIS_ORIG_MULTIPLIER_POSTPROCESSING # manual post processing
        results.append((ranking_score, doc_id))

    # Step 7: Sort the results in descending order of score
    results.sort(key=functools.cmp_to_key(comparator))
    return results

def find_term_specific_weight_for_specified_id(doc_id, posting_list):
    """
    Returns the accumulated weight (regardless of field type) for the stated document of doc_id seen in posting_list (which is a PostingList for a given dictionary term)
    Score is returned in ltc scheme, following that for query
    """

    result = 0 # remains 0 if the doc_id marked relevant does not contain the term that the PostingList represents for
    tf = 0

    # scan through posting_list and accumulate to get the specified document's total tf regardless of field type
    for posting in posting_list.postings:
        if (posting.doc_id == doc_id):
            # number of positions in positional index is the number of occurrences of this term in that field
            tf += len(posting.positions)

    # if the specified document does contain the term, return ltc weight (following query), otherwise return 0
    if (tf > 0):
        df = posting_list.unique_docids
        N = len(ALL_DOC_IDS)
        result = (1 + math.log(tf, 10)) * math.log(N/df, 10)

    return result

def obtain_all_cos_score_terms(relevant_docids, tokens_arr):
    """
    Returns a set of all terms which are accumulated from
    all relevant documents' top_K terms, and the processed version of those in tokens_arr
    """
    res = []
    for impt in relevant_docids:
        ls = ALL_DOC_IDS[impt]
        for t in ls:
            res.append(t)
    processed_terms = [stem_word(w.strip().lower()) for w in tokens_arr]
    for t in processed_terms:
        res.append(t)
    return set(res) # all unique now, all are processed

def remove_term_processed_from_set(term, union_of_relevant_doc_top_terms):
    """
    Removes the processed version of the term from a set that stores this processed term
    """
    processed_term = stem_word(term.strip().lower())
    if processed_term in union_of_relevant_doc_top_terms:
        union_of_relevant_doc_top_terms.remove(processed_term)


def calculate_relevant_centroid_weight(relevant_docids, posting_list):
    """
    Calculates the averaged tf-idf value (ltc weighting scheme) for all the relevant documents for the relevant documents' 'centroid'
    Each ltc value of a document is adjusted for distribution with respect to the term's occurrence in that document
    Note: posting_list here represents a particular term, and contains Postings of documents with specific zones/types
    """
    accumulated_value = 0
    for doc_id in relevant_docids:
        # divide by doc_lengths for effective normalisation to consider distribution of the current term within the document
        accumulated_value += find_term_specific_weight_for_specified_id(doc_id, posting_list)/DOC_LENGTHS[doc_id]
    return accumulated_value/len(relevant_docids)

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
    Returns the list representation (postings) of the term's PostingList
    or an empty list if no such term exists in index
    """
    # NOTE: LOWERCASING IS ONLY DONE HERE.
    term = term.strip().lower()
    term = stem_word(term)
    if term not in D:
        return None
    POSTINGS_FILE_POINTER.seek(D[term])
    return pickle.load(POSTINGS_FILE_POINTER)

def find_already_processed_term(term):
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
    L1 = len(list1.postings)
    L2 = len(list2.postings)
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
    #Take in list of original, unexpanded query terms
    #Calculate the query_term_weight of the individual terms from the original query
    #Query terms with high weight >= 1.2, would be further expanded and their synonyms will be added to the original term
    #Process the expanded list of query terms

    term_frequencies = Counter(terms)
    expanded_terms = []
    for t in terms:
        expanded_terms.append(t)
        # this is the same way posting list for individual phrases/words have been obtained in cosine_score
        # Weight for individual queries needs to be measured here as well in order to check which quey words/ phrases are the more important ones and
        # are worth expanding
        if " " in t:
            posting_list = perform_phrase_query(t)
        else:
            posting_list = find_term(t)

        if posting_list is None:
            continue

        query_term_weight = get_query_weight(posting_list.unique_docids, term_frequencies[t])
        if query_term_weight >= 1.2 :
            expanded_terms.extend(query_expansion(t, terms))

    expanded_terms = process(expanded_terms)
    res = cosine_score(expanded_terms, relevant_docids)
    return res

def split_query(query):
    """
    split_query extracts out phrases and terms from the unedited first line of the query file
    Note: Phrases for phrasal queries are identified by " double inverted commas,
    which are removed in the process of creating these phrases
    """
    start_index = 0
    current_index = 0
    is_in_phrase = False
    is_boolean_query = False
    terms = []

    while current_index < len(query):
        current_char = query[current_index]
        if current_char == "\"":
            # This is the start or end of a phrasal query term
            # Note that this phrasal query is treated like a free-text query, but on a fixed term
            # We will differentiate them later on
            if is_in_phrase:
                is_in_phrase = False
                terms.append(query[start_index:current_index]) # entire phrase as a term
                start_index = current_index + 1 # +1 to ignore the space after this
            else:
                start_index = current_index + 1
                is_in_phrase = True
        elif current_char == " ":
            # this is the end of a non-phrasal query term, can append directly
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

def query_expansion(query, unexpanded_tokens_arr):
    #Take in a word from the query
    #Expand on the synonyms of the word
    #return the set of synonyms

    stop_words = set(stopwords.words('english'))

    syn_set = set()

    for s in wordnet.synsets(query):
        for l in s.lemmas():
            if l.name() not in unexpanded_tokens_arr:
                syn_set.add(l.name())

    return syn_set
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
    D = pickle.load(dict_file_fd) # dictionary with term:file cursor value entries
    DOC_LENGTHS = pickle.load(dict_file_fd) # dictionary with doc_id:length entries
    ALL_DOC_IDS = pickle.load(dict_file_fd) # dictionary with doc_id:top_K terms (for optimisation, e.g. Rocchio Algo)
    POSTINGS_FILE_POINTER = open(postings_file, "rb")
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
