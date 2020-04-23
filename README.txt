This is the README file for A0183871W-A0173165E-A0136174H-A0171638Y's submission

== Python Version ==

We're using Python Version 3.6.8 for this assignment.

== General Notes about this assignment ==

***INDEXING PHASE***

During the indexing phase, we firstly read the single csv file to separate it into individual legal documents. These individual documents are represented by
a dictionary with keys ('doc_id', 'title', 'content', 'date_posted', 'court') using the immediate information available in the csv file. These document-specific 
values are then used to generate positional indexes for title, content, date_posted, and court, with the terms of these positional indexes obtained via the 
generate_positional_indexes function, which first tokenises, filters punctuations, casefolds to lowercase, and then stems words appearing in the respective 
fields/zones. Once again, these positional indexes, which are specific to a particular document, will be stored in the document's dictionary. We repeat this
to obtain a dictionary representation for all documents, and store all documents in set_of_documents. Moreover, we use all the positional indexes to 
accumulate the counts and identify the top K (K is an arbitrary number specified at the top of the index.py file) and store them in the last dictionary entry 
document[top_K] for every document. This is to facilitate Rocchio Algorithm Query Refinement later on during searches.

Once the set_of_documents contains all documents, we accumulate in tokens_list all the positional indexes for each term, regardless of which zone/field it 
appears as. The tokens_list is sorted by term, then doc_id, then by title, court, date_posted, and finally content. Once we have all this information sorted,
we can now begin creating PostingLists for each term and fill them up with their relevant Postings. Note that here, all Postings are stored in the 
term-specific PostingList regardless of which zone/field it appears in. Also note that each Posting represents a doc_id, the zone/field of the Posting, and 
the positional index showing all positions of the term in that document's specific zone/field.

Next, as with previous homeworks, we store and write all the useful information. The postings file postings.txt will contain all the PostingLists, while the
dictionary file dictionary.txt is used to store the dictionary containing term:file cursor values, a dictionary containing all document lengths, and the
dictionary that stores the top K terms for each document (this is used for Rocchio Algorithm later on).

***SEARCH PHASE***

We begin by loading the previously stored data: the PostingList file cursor values, all document lengths for subsequent normalisation, and all doc_ids with their 
top K most common terms, which will be used for optimisation. We will then parse the query file, with the first line being the actual query and subsequent lines 
(which represent documents marked as relevant by the law experts) into an array relevant_docids. We will now begin search.

Firstly, the parse_query function takes in the query and calls split_query to obtain words to process into terms later on. This process also determines the search 
type. Here, in split_query, we are splitting the original query given from the query file into either words of length 1, or phrases (identified by double 
inverted commas in a phrasal query). In the process, if we encounter the Boolean Retrival keyword "AND", we know this is a boolean query and set is_boolean_query 
to True. Otherwise, we will process this term in a free-text query. Please note that phrasal queries will be performed as part of processing either boolean 
and/or free-text queries (otherwise, we will assume it is a free-text query first, then process it as a phrasal query). A phrasal query can easily be 
identified by spaces (" ") in its term, because split_query produces the respective phrases whenever it encounters the double inverted commas. At the end of
this function, we will have a sequence of words and phrases. Once done, the free-text or boolean query (which contains any phrasal queries) is then executed.

DIFFERENT TYPES OF SEARCH QUERIES:

Note: Phrasal queries are done by finding PostingLists for each individual term, and then performing an AND merge. This way, we are able to get the final 
documents containing all the terms in the phrase. In our program, since phrasal queries may occur in a mixture of queries with free-text queries or boolean
queries, we classify them (by default as free-text, otherwise with whatever it is in a mixture of queries with) and run them together with those queries.

1. Free-text queries

For free-text queries, query expansion is implemented. We first take in the list of words/ phrases in the query terms and measure the query term weight of 
each individual words/phrases. If the weight is more than or equal to a particular threshold, query expansion is done on it. This is to avoid query expansion
on every word and phrases, and only to be done on the important words. Once the query is expanded, the list of words/phreases are then processed.

We process the words/phrases into final index terms by filtering through punctuations and removing some of them like apostrophes. As this process can 
possibly generate additional unneeded spaces, we will then remove these unnecessary spaces to prevent them from being detected as a term. Next, we will 
perform scoring and ranking, and possibly query refinement via the Rocchio Algorithm if needed. Note that here, we have knowledge of the documents that the
law expert marked as relevant, so we can perform the query refinement for free-text queries which are not entirely phrasal queries.

To facilitate the Rocchio Algorithm, we use the previously obtained list of top K most common terms of each relevant-marked document, and obtain their union, along
with those in the current query. This is done regardless of whether Rocchio is actually performed, as there is a high chance that the query is not entirely phrasal. 

Next, we will go through all the query terms to calculate score contributions first, before moving on to those in the union but are not in the query. This happens in
the following manner:

1. Obtain PostingList for the current term, performing a phrasal query if the current term is a phrase. 
2. If there is no such PostingList, this term is an invalid term and we move on to the next term. Else, if phrasal query gives an empty PostingList, there is zero score
contribution so we simply move on to the next term also.
3. Calculate the ltc weight for the current query term.
4. If Rocchio Algorithm is used for query refinement, remove the current term from the union set of accumulated terms. Perform the query refinement by calculating the
centroid of the relevant-marked documents for this current term. To do this, since the PostingList may contain several Postings belonging to the same document, we will
consider all the occurrences throughout the document (regardless of field/zone) and then calculate their ltc scores (to be consistent with the ltc scheme for query vector)
before normalising (to consider the term distribution within the document), dividing this accumulated value by the individual document's length. We then accumulate all 
these normalised values, and average them by dividing by total number of relevant-marked documents. If this value is positive and non-zero, we add it to let it influence 
the initial query's value for this current term. Here, we use EMPHASIS_ON_ORIG and EMPHASIS_ON_RELDOC to determine how much the refined query's value will be influenced 
by the original query value and the relevant documents' value, respectively.
5. Calculate/Accumulate the score contribution from the current term for the documents that contain this current term. Here, we use the lnc.ltc scheme. Moreover, as we
recognise the importance of fields, each field will have a multiplier attached to it to yield a differently-scaled score contribution.
6. Repeat the above for all the terms in the original query, removing the query's term from the unioned set each time.
7. If the query is not entirely phrasal, we will perform Rocchio Algorithm for the remaining terms in the unioned set. Once again, we will remove (from the unioned set) 
the term we are currently considering for score contributions to documents each time. Note that the main difference here, compared to the terms previously considered which
appear in the query, is that the query's inital value is 0 because it does not contain these terms. So, the score contribution is now derived from the relevant-marked 
documents' centroid values, and not from the initial query (which has a value of 0 for this term). We will calculate the score contributions for all these terms, 
eventually updating document scores for all documents that contain terms once in this unioned set of terms.
8. Finally, we will perform normalisation on the lnc.ltc scores obtained from all the above score contributions, and perform some post-processing to emphasise documents
which contain terms in the initial query. Documents which are relevant will therefore be shown with the highest scores first.


2. Boolean queries

Meanwhile, for boolean queries, only "AND" operatore are supported. Hence, "AND" operation is conducted on all the words/phrases that appear in boolean query. 
Firstly, all "AND" keyword are removed from the query. Similiarly, we process the words/phrases into final index terms by filtering through punctuations. We then obtain 
the postinglist for the word/phrase. 

It is to note that when processing phrases, because gap encoding is used to store the position of the posting, we would need to add the previous values to obtain the 
actual positional index. Then, the actual positional indexes are compared to ensure that the query terms are next to each other.

Once the postinglist for the word/phrase is obtained, they are then merged to obtain a postinglists where all the query terms appears in the listed document. 
The final posting list are then ranked. Scores are added for every query term present in the document. The score are dependent on which field the term appears in. 
Terms that are the doc_id have higher score, followed by title, court then content. After the scores are tallied, the document with the highest score are placed first in the output.



EXPERIMENTS

We have quite a few arbitrary values: 
- K (index.py) determines how many most common terms of a document do we store in the index, which will then affect how many terms we will use for the Rocchio Algorithm
- EMPHASIS_ON_ORIG (search.py) determines how much emphasis we place on the original query's value for a particular term's score in the Rocchio Algorithm
- EMPHASIS_ON_RELDOC (search.py) determines how much emphasis we place on the relevant-marked documents for a particular term's score in the Rocchio Algorithm
- EMPHASIS_ORIG_MULTIPLIER_POSTPROCESSING multiplies the score for terms that appear in the original query, which we understand to be more significant
- There are also multipliers for the type of field/zone that the term appears in, which affects the score contribution of these Postings for their associated document.

We have played around and varied them to try to optimise our search results, but due to the limited number of times that the leaderboards is generated (because it takes
rather long to generate), these values may not be the most optimal. Still, we have tried on our end to find optimal values.

Workload

The team split work objectively, with one person working on overall architecture (such as the functional skeleton), one on Rocchio Algorithm, one on Query Expension via
Manual Thesaurus, and one on documentation. Moreover, we overlap and check on one anothers' parts.

== Files included with this submission ==

index.py - the file to initiate indexing
search.py - the file containing rules on how to perform each search
dictionary.txt - the generated dictionary containing the term to file cursor of PostingList mappings, all document lengths, and the term to top K term mappings.
postings.txt - the file containing all the PostingLists for all the terms
BONUS.docx - the file containing explanation on our query expansion/refinement techniques

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[X] We, A0183871W-A0173165E-A0136174H-A0171638Y, certify that we have followed 
the CS 3245 Information Retrieval class guidelines for homework assignments.  
In particular, we expressly vow that we have followed the Facebook rule in 
discussing with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

We suggest that we should be graded as follows:

We suggest that we should be marked more with regard to our actual techniques, and not so much on the arbitrary values we chose (which are largely 
experimental) that will give different ranking results.

== References ==

Not Applicable
