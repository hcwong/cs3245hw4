This is the README file for A0183871W-A0136174H's submission

I am A0183871W. My email is e0310666@u.nus.edu
I am A0136174H. My email is e0321481@u.nus.edu

== Python Version ==

We're using Python Version 3.76 for this assignment.

== General Notes about this assignment ==

1. Building VSM
We begin by building the VSM containing all the Postings. Here, one dictionary of PostingLists, 
one dictionary of document lengths (doc_lengths), and one list of document ids (doc_ids) are also created.
Firstly, we use the get_words() function obtain a list of tuples (doc_id, Counter(all of doc_id's {term: term frequency} entries)) 
from all documents in the directory. Here, the process of filtering for punctuations, case-folding to lowercase, tokenisation and stemming
are performed. Then, we will flatten this, and sort the entries such that we have a sorted list 
of [term, (doc_id, freq_in_doc)] elements, stored in tokens_list. This sorting is by term, then by increasing doc_id.
Secondly, obtain the list of all document ids in the directory.
Thirdly, since we have the required sorted data to begin building the index, we begin doing so, creating a new PostingList for each term
and then populating each PostingList with Postings made using (doc_id, freq_in_doc) from tokens_list. It should be noted that we pre-compute
the weight self.weight = 1 + math.log(freq, 10) here for every Posting to optimise the search phase later on.
Lastly, we compute the length of each document vector for normalization during the search stage

2. Writing the VSM and relevant data to storage
We write the list of all doc_ids at the start of the postings file, followed by the PostingLists themselves, with their file cursor locations
stored in a dictionary d which contains mappings of each term to its respective file cursor value of its PostingList.
Since d is important to know where to read the PostingLists from, we write it into the dictionary file, followed by writing the dictionary of
all document lengths, which is used for normalisation during the search phase.

3. Reading from storage to begin Search phase
We open both dictionary file and postings file to load the dictionary containing all mappings of each term to its respective file 
cursor value of its PostingList into D, and Document Lengths into DOC_LENGTHS, and also read in all Document IDs.
Thereafter, we open the queries file to read and parse queries line by line, each line representing a query. Once again, the process 
of filtering for punctuations, case-folding to lowercase, tokenisation and stemming are performed to obtain dictionary terms.
Once we have all processed dictionary terms, we run the cosine_score function on these query terms to obtain the top 10 documents in terms
of scores with respect to the query terms. This relies on cosine similarity score contributions, which depends on whether the document contains
the query term:
- For every query term, non-zero cosine similarity contributions are made only for documents containing the query term. Therefore, to optimise, we
only do calculation for these documents and do so pointwise. 
- Here, we obtain score contributions term-wise: accumulate them for individual documents containing the term before moving onto the next document,
to further calculate and add more cosine similarity score contributions; once done, move onto the next term. 
We do so whenever we encounter a new term rather than only after constructing the query vector, to save on overhead.
- Weights for documents are lnc calculations previously calculated during the indexing phase, to optimise speed.
- We normalise at the end by only dividing by the document length (and not the query vector length) to optimise speed. This is since the query vector
length is the same for all the results and hence dividing by it will not affect comparison. Hence, this saves some operational costs.

4. Obtaining and Writing Search Results
Once we obtain each result, we store the tuple (ranking_score, doc_id) into a max heap data structure, which sorts by decreasing score and then increasing
doc_id. Here, we simply call functions to obtain the top 10 (or less, if all documents exhausted) relevant documents as required by the Homework. 
This is followed by writing these into the same line as they represent results for a single query.

Experiments:

We eventually settled to use the above implementation, which is reflected in our code. Below describes some experiments we did, but decided not to
proceed with their implementations.

We opted to filter for punctuation as we felt that the common ones such as parantheses, commas and fullstops should not be treated as separate terms 
as it would negatively impact the accuracy of search.
Also, during free text search, such special characters are not often seen and hence we opted to process them in the corpus.
For google search however, it may not be wise to remove these characters from the query as they have special meaning, but for the purposes of this 
assignment, we felt that it was okay.

We tried using Champion Lists, and somewhat mixing the idea of impact-ordered postings where upon crafting Champion Lists of a certain defined size,
we would only calculate score contributions from these documents in the Champion Lists. However, the results given were largely different from those
shared by our classmates on the Luminus Forum, with only about 30% match. The reason for this could be that the contributions for documents not in
these Champion Lists were non-zero and effectively significant, and having more search terms could imply these significant score contributions add up
and therefore lead to different document rankings. Here, our Champion List sizes included taking the larger of 30 or 20% of the term's PostingList size,
and our second try, the larger of 10 or 10% of the term's PostingList size. As the Champion List size grew smaller, while there are indeed fewer documents
to calculate the scores for, and hence fewer operations, the accuracy reduced (comparing with the Luminus Forums). True enough, when we increased it to
take the larger of 100 or 30% of the term's PostingList size, accuracy did indeed improve to about 75% similarity with our classmates' results. Eventually,
we decided that correctness was more important than speed, considering that we have already did some optimisation on our end and that the searches are
fast enough - and especially so with the somewhat considered small Reuters data we are givenm where most of the posting lists tend to be less than 30.

We also tried to think of ways to implement Index Elimination based on only considering terms with high idf and combine this with Champion Lists, but decided
against it due to the lnc calculation requirement for documents.


== Files included with this submission ==

index.py implements the indexing phase
search.py implements the search phase with some optimisation
dictionary.txt shows the pickled representation of the dictionary containing the term to file cursor value mappings and dictionary of all document lengths
postings.txt shows the pickled representation of all doc_ids and PostingLists
README.txt is this file explaining our solutions and administrative details


== Statement of individual work ==

[x] We, A0183871W-A0136174H, certify that we have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, we
expressly vow that we have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.

== References ==

Luminus Forum Post on the results of some student generated queries.
