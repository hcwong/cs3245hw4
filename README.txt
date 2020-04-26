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
accumulate the counts and identify the top K terms for each document (K is an arbitrary number specified at the top of the index.py file) and store them in the 
dictionary entry document[top_K] for every document. This is to facilitate Rocchio Algorithm Query Refinement later on during searches.

To save on indexing space, we also employ gap encoding and variable byte encoding for positional indices. We first gap encode everything, and afterwards use an
external file/library to do variable byte encoding.

Once the set_of_documents contains all documents, we accumulate in tokens_list all the positional indexes for each term, regardless of which zone/field it 
appears as. The tokens_list indexes are sorted by term, then doc_id, then by title, court, date_posted, and finally content. Once we have all this information 
sorted, we can now begin creating PostingLists for each term and fill them up with their relevant Postings. Note that here, all Postings are stored in the 
term-specific PostingList regardless of which zone/field it represents, as long as they are the same dictionary term. Also note that each Posting represents 
a doc_id, the zone/field of the Posting, and the positional index showing all positions of the term in that document's specific zone/field.

Next, as with previous homeworks, we store and write all the useful information. The postings file postings.txt will contain all the PostingLists, while the
dictionary file dictionary.txt is used to store the dictionary containing term:file cursor values, a dictionary containing all document lengths, and the
dictionary that stores the top K terms for each document (this is used for Rocchio Algorithm later on).

***SEARCH PHASE***

We begin by loading the previously stored data: the PostingList file cursor values, all document lengths for subsequent normalisation, and all doc_ids with their 
top K most common terms, which will be used for optimisation. We will then parse the query file, with the first line being the actual query and subsequent lines 
(which represent documents marked as relevant by the law experts) into an array called relevant_docids. We will now begin search.

Firstly, the parse_query function takes in the query and calls split_query to obtain words to process into terms later on. This process also determines the search 
type. Here, in split_query, we are splitting the original query given from the query file into either words of length 1, or phrases (identified by double 
inverted commas in a phrasal query). In the process, if we encounter the Boolean Retrival keyword "AND", we know this is a boolean query and set is_boolean_query 
to True. Otherwise, we will process this term in a free-text query. 

Please note that phrasal queries (appearing with "AND") will be performed as part of a boolean query. However, if the query is purely phrasal, we will process it 
as a free-text query, which will then be processed as a purely phrasal query. In other words, if the entire query contains only one phrase and nothing else, it 
will be processed as part of free-text query. If the entire query contains more than one phrase, or contains a phrase combined with "AND" keyword(s), then it is 
considered a boolean query. A phrasal query can easily be identified by spaces (" ") in its term, because the split_query function produces the respective phrases 
whenever it encounters the double inverted commas. At the end of this function, we will have a sequence of words and phrases. Once done, the free-text or boolean 
query (which contains any phrasal queries) is then executed.

DIFFERENT TYPES OF SEARCH QUERIES:

Note: Phrasal queries are done by finding PostingLists for each individual term, and then performing an AND merge. This way, we are able to get the final 
documents containing all the terms in the phrase. In our program, since phrasal queries may occur in a mixture of queries with free-text queries or boolean
queries, we classify them appropriately in our program (as mentioned above) and run them together with those queries.

1. Free-text queries

For free-text queries, query expansion is implemented. We first take in the list of words/phrases in the query terms and measure their weight of to identify the
more significant terms that we should expand on. If the weight is more than or equal to a particular threshold, query expansion is done on it. This avoids 
unnecessary query expansion, which can affect the performance/accuracy of our results. Once the query is expanded, we have a finalised list of words/phrases to
process for the free-text query.

We process the words/phrases into final index terms by filtering through punctuations, replacing some of them with spaces and removing some of them (e.g. 
apostrophes). As this process can possibly generate additional unneeded spaces, we will remove these unnecessary spaces to prevent them from being detected as 
a term. Next, we will perform scoring and ranking, and possibly query refinement via the Rocchio Algorithm if needed. Note that here, with knowledge of the 
which documents are relevant, we can perform the query refinement for free-text queries which are not entirely phrasal queries (a query with only one phrase).

To facilitate the Rocchio Algorithm, we use the previously obtained list of top K most common terms of each relevant-marked document, and obtain their union
with those in the current query. This is done regardless of whether the Rocchio Algorithm is actually performed, as there is a high chance that the query is 
not entirely phrasal. As the Rocchio Algorithm is performed term-wise, the corresponding terms are removed accordingly.

Next, we will go through all the query terms to calculate score contributions first, before moving on to those in the union but are not in the query. This happens 
term-wise in the following manner:

1. Obtain PostingList for the current term, performing a phrasal query if the current term is a phrase. 
2. If there is no such PostingList, this term is an invalid term and we move on to the next term. Else, if phrasal query gives an empty PostingList, there is a 
score contribution of zero so we simply move on to the next term.
3. Calculate the ltc weight for the current query term.


(If Rocchio Algorithm is performed)
4. If Rocchio Algorithm is used for query refinement, remove the current term from the unioned-set of accumulated terms to mark it as done. Perform the query 
refinement by calculating the centroid of the relevant-marked documents for this current term. To do this, since the PostingList may contain several Postings 
belonging to the same document (but of different zones/fields), we will consider all the occurrences throughout the document (regardless of field/zone) and then 
calculate their ltc scores (consistent with ltc scheme for query vector) before dividing this by their document length (to consider the term distribution within
the document). We then accumulate all these values and average them by dividing by total number of relevant-marked documents to find the centroid's value. If this 
value is positive and non-zero, we add it to let it influence the initial query's value for this current term. Here, we use EMPHASIS_ON_ORIG and EMPHASIS_ON_RELDOC
to determine how much the refined query's value will be influenced by the original query value and the relevant documents' value, respectively.


5. Calculate/Accumulate the score contribution from the current term for the documents that contain this current term. Here, we use the lnc.ltc scheme. Moreover, 
as we recognise the importance of fields, each field will have a multiplier attached to it to yield a differently-scaled score contribution.
6. Repeat the above for all the terms in the original query, removing the query's term from the unioned set each time.


(If Rocchio Algorithm is performed)
7. If the query is not entirely phrasal, we will perform Rocchio Algorithm for the remaining/untouched terms in the unioned set. Once again, we will remove the 
current term each time to mark it as done. Note that the main difference here, compared to the terms which appear in the query, is that the query's inital value 
is 0 because it does not contain these terms. So, the score contribution is now derived entirely from the relevant-marked documents' centroid values. This is done
for all the terms in the unioned set.


8. Finally, once we have all the accumulated scores, we will perform normalisation on these lnc.ltc scores obtained from all the above score contributions, and do
some post-processing (using a multiplier) to emphasise documents which contain terms in the initial query. Documents which are relevant will therefore be ranked 
higher.


2. Boolean queries

Meanwhile, for boolean queries, only "AND" operators are supported. Hence, "AND" operation is conducted on all the words/phrases that appear in boolean query. 

Firstly, all "AND" keywords are removed from the query to identify every query term/phrase. We process these words/phrases into final index terms by filtering 
through punctuations (as with free-text queries). We then obtain the PostingList for each processed word/phrase. 

Once the PostingLists for each word/phrase is obtained, they are then merged to obtain a PostingList where all the query terms appear in the listed document. 
The final posting list are then ranked. Scores are added for every query term present in the document with different fields/zones having a different weightage in scoring. 
Terms appearing as doc_id have higher weighting, followed by title, court then content. After the scores are tallied, documents with the highest scores are shown first,
and those with lowest scores shown last in the output file.

It is important to note that when processing phrases, because gap encoding is used to store the position of the posting in the positional index, we need to repeatedly
add values (representing these gaps) to the previous position to obtain the actual positions. Then, the relevant positions are compared to ensure that the query terms
are indeed next to each other. The merging of these positions is done as per the lecture, where we look for terms that have consecutive positions as with the original
query. For the purposes of this project, we only merge the positions if the fields/zones and the doc_id of the Posting is the same, because the positional indices 
use gap encoding and each positional index is for a different kind of fields/zones.

However, because of the nature of intersection operations, it is highly likely that we will not get any results from the strict boolean query. This problem is further 
compounded by Prof Jin's post on forum that the boolean match must be exact.
If, after a boolean query, we have fewer results than a stated threshold, we will then perform an OR merge for the terms separated by "AND".
If still we do not have enough results, then we break down the phrasal query into individual word terms (no longer a phrase) and perform free-text query with Rocchio
Algorithm (using the relevant doc ids provided by the judge) to get a list of likely documents that match it. However, we weight these documents less than the boolean 
result, as the boolean result is likely to be rarer, and that this is a backup measure. We then merge all our results and return them accordingly.

EXPERIMENTS

We have quite a few arbitrary values: 

- K (index.py) determines how many most common terms of a document do we store in the index, which will then affect how many terms we will use for the Rocchio Algorithm.
It seems that K value of around 12-14 gives better results based on the leaderboard.
- EMPHASIS_ON_ORIG (search.py) determines how much emphasis we place on the original query's value for a particular term's score in the Rocchio Algorithm
- EMPHASIS_ON_RELDOC (search.py) determines how much emphasis we place on the relevant-marked documents for a particular term's score in the Rocchio Algorithm.
It seems placing a lower significance on relevant documents and higher on original query may give better performance.
- EMPHASIS_ORIG_MULTIPLIER_POSTPROCESSING multiplies the score for terms that appear in the original query, which we understand to be more significant.
This is proven to help in ranking document IDs containing the query term higher.
- There are also multipliers for the type of field/zone that the term appears in, which affects the score contribution of these Postings for their associated document.

We have played around and varied them to try to optimise our search results, but due to the limited number of times that our index, and the leaderboard can be generated, 
these values may not be the most optimal. Still, we have tried our best on our end to find optimal values.

Workload

The team split work objectively, with one person focusing on overall architecture (such as the functional skeleton), one on Rocchio Algorithm, one on Query Expension via
Manual Thesaurus, and one on documentation. Moreover, we overlap and check on one anothers' parts.

== Files included with this submission ==

index.py - the file to guide the indexing phase.
search.py - the file containing rules on how to perform each search.
dictionary.txt - the generated dictionary containing the term to file cursor of PostingList mappings, all document lengths, and the term to top K term mappings.
postings.txt - the file containing all the PostingLists for all the terms.
encode.py - This is the external file we use to do variable byte encoding. The source is acknowledged at the top of the file.
BONUS.docx - the file containing explanation on our query expansion/refinement techniques.

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

https://github.com/utahta/pyvbcode/blob/master/vbcode.py - Variable byte encoding code
