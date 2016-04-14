#!/usr/bin/python
import re
import nltk
import sys
import getopt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stemmer = nltk.stem.porter.PorterStemmer()

from math import log10, sqrt

import xml.etree.ElementTree as ET

pseudo_relevance_threshold = 10

# ============================
# Initialisation functions
# ============================

def build_dict(input_dict_file):
    """
    Builds the dictionary from the dictionary file. Kept in memory.
    Returns the following dictionaries as a 4 tuple:

    term_dict:          Containing a dictionary of terms, with pointers to their postings
                        lists in the postings file and their idf.
    docs_dict:          Containing a dictionary of document ids, with pointers to their 
                        lists of terms in the postings file, and their total term count.
    doc_fields_dict:    Containing a dictionary of document ids, with pointers to their
                        lists of relevant document ids.
    ipc_indices:        Containing a dictionary of dictionaries, one mapping each document
                        id in the collection to their corresponding IPC categories, and other
                        term based dictionaries of each of the types of IPC categories (IPC
                        Classes, IPC Subclasses and IPC Groups). Each of these dictionaries
                        are similar to 'term_dict', but are treating IPC categories as
                        document ids.

    """
    mode = -1 # 0: term_dict, 1: doc_dict, 2: doc_fields_dict, 3: doc_id_to_ipc, 4,5,6: term to ipc categories
    dict_file = file(input_dict_file, 'r')
    term_dict = {} # key: term, value: (byte_offset, idf)
    docs_dict = {} # key: doc_id, value: (byte_offset, total_term_count)
    doc_fields_dict = {} # key: doc_id, value: byte_offset

    doc_id_to_ipc = {}
    term_to_ipc_class = {}
    term_to_ipc_subclass = {}
    term_to_ipc_group = {}

    for line in dict_file:
        line = line.strip()
        split_line = line.split(" ")
        if line == "## Term ##":
            mode = 0
        elif line == "## Doc ID ##":
            mode = 1
        elif line == "## Fields ##":
            mode = 2
        elif line == "## Doc ID to IPC ##":
            mode = 3
        elif line == '## Term to IPC Class ##':
            mode = 4
        elif line == '## Term to IPC Subclass ##':
            mode = 5
        elif line == '## Term to IPC Group ##':
            mode = 6
        elif mode == 0:
            term_dict[split_line[0]] = (int(split_line[1]), float(split_line[2]))
        elif mode == 1:
            docs_dict[split_line[0]] = (int(split_line[1]), int(split_line[2]))
        elif mode == 2:
            doc_fields_dict[split_line[0]] = int(split_line[1])
        elif mode == 3:
            doc_id_to_ipc[split_line[0]] = (split_line[1], split_line[2], split_line[3])
        elif mode == 4:
            term_to_ipc_class[split_line[0]] = (int(split_line[1]), float(split_line[2]))
        elif mode == 5:
            term_to_ipc_subclass[split_line[0]] = (int(split_line[1]), float(split_line[2]))
        elif mode == 6:
            term_to_ipc_group[split_line[0]] = (int(split_line[1]), float(split_line[2]))

    ipc_indices = {"doc_ids": doc_id_to_ipc, "ipc_class": term_to_ipc_class, "ipc_subclass": term_to_ipc_subclass, "ipc_group": term_to_ipc_group}

    dict_file.close()
    return (term_dict, docs_dict, doc_fields_dict, ipc_indices)

def execute_query(input_post_file, input_query_file, output_file, term_dict, docs_dict, doc_fields_dict, ipc_indices):
    """
    Tests the query in the input_query_file based on the dictionary and postings.
    Writes results into output_file.
    """   

    # Initialisation
    tree = ET.parse(input_query_file)
    root = tree.getroot()
    postings = file(input_post_file, 'r')
    output = file(output_file, 'w')

    # Treat query as a string (concatenate title and description)
    # Also removes "Relevant documents will describe" from start of description
    query = root[0].text.strip() + " " + root[1].text.strip()[33:].strip()

    # Score and rank the IPC Classes found in the corpus
    ipc_class_scores = vsm_query(query, ipc_indices["ipc_class"], postings)
    ipc_class_min_score = min(ipc_class_scores, key=lambda x: x[1])[1]
    ipc_class_scores = dict(ipc_class_scores)

    # Score and rank the IPC Subclasses found in the corpus
    ipc_subclass_scores = vsm_query(query, ipc_indices["ipc_subclass"], postings)
    ipc_subclass_min_score = min(ipc_subclass_scores, key=lambda x: x[1])[1]
    ipc_subclass_scores = dict(ipc_subclass_scores)

    # Score and rank the IPC Groups found in the corpus
    ipc_group_scores = vsm_query(query, ipc_indices["ipc_group"], postings)
    ipc_group_min_score = min(ipc_group_scores, key=lambda x: x[1])[1]
    ipc_group_scores = dict(ipc_group_scores)

    ##############################
    # Not used in Final Submission
    # Perform Language Model Query
    # LM_results = [x[0] for x in language_model_query(query.strip(), docs_dict, term_dict, postings)][:pseudo_relevance_threshold]
    ##############################

    # Perform VSM query
    vsm_scores = vsm_query(query.strip(), term_dict, postings)
    
    ##############################
    # Not used in Final Submission
    # vsm_results = map(lambda x: x[0], vsm_scores)
    # Combine initial query
    # initial_result = vsm_results[:pseudo_relevance_threshold] # list(set(LM_results) | set(vsm_results))
    # results = rocchio_algorithm_query(initial_result, query, term_dict, docs_dict, doc_fields_dict, postings)
    ##############################

    # Weight each document based on their IPC Categories
    doc_id_to_ipc = ipc_indices["doc_ids"]
    updated_results = []
    for result in vsm_scores:
        # Find document's corresponding IPC Scores
        doc_id = result[0]
        old_score = result[1]
        class_score = ipc_class_min_score
        subclass_score = ipc_subclass_min_score
        group_score = ipc_group_min_score
        if doc_id in doc_id_to_ipc:
            (ipc_class_name, ipc_subclass_name, ipc_group_name) = doc_id_to_ipc[doc_id]
            class_score = ipc_class_scores[ipc_class_name]
            subclass_score = ipc_subclass_scores[ipc_subclass_name]
            group_score = ipc_group_scores[ipc_group_name]
        # Weight Document's VSM Score with IPC Category Score
        new_score = old_score * ((class_score * subclass_score * group_score)**2)
        updated_results.append((doc_id, new_score, old_score))
    updated_results.sort(key=lambda x: (-x[1], -x[2]))

    # Write results to output file
    output_line = reduce(lambda x, y: x + str(y[0]) + " ", updated_results, "").strip()
    output.write(output_line)

    output.close()

# ============================
# Language Model querying
# ============================

################################
# Not used in Final Submission #
################################
def language_model_query(query, docs_dict, term_dict, postings):
    scored_docs = {}
    probability = 1 # P(t|d)
    lambda_weight = 0.5 # hard-coded, probably need to find the optimal weight based on how long the query is

    collection_length = get_collection_length(docs_dict)

    for token in word_tokenize(query):
        token = stemmer.stem(token).lower()

        # To remove stop words and punctuation from the query (since neither of which are indexed)
        if token not in term_dict:
            continue

        # For each token, find the list of documents in which the term appears and the token's number of occurrences
        doc_term_frequency = build_doc_term_frequency(token, term_dict, postings) # a dictionary
        docs = [k for k,v in doc_term_frequency.items()]
        collection_term_frequency = sum([v for k,v in doc_term_frequency.items()])

        for doc in docs:
            # Formula: weight * n(token appears in doc)/doc_length + (1-weight) * n(token appears in collection)/collection_length
            term_frequency = doc_term_frequency[doc]
            document_length = get_document_length(docs_dict, doc)

            probability = lambda_weight * term_frequency/document_length + (1-lambda_weight) * collection_term_frequency/collection_length
            if doc in scored_docs:
                scored_docs[doc] *= probability
            else:
                scored_docs[doc] = probability

    # print scored_docs
    output = [(k, v) for k, v in scored_docs.items()]
    output.sort(key=lambda x: x[1], reverse=True)
    return output
    # return sorted([(k,v) for k, v in scored_docs.items()], lambda x: x[1], reverse=True)

################################
# Not used in Final Submission #
################################
def build_doc_term_frequency(token, term_dict, postings):
    reader = get_term_postings_reader(token, term_dict, postings)
    dct = {} # { doc_id: raw_term_frequency, ... }
    while True:
        # next() return format: (doc_id, raw_term_frequency, normalized_log_weighted_term_frequency)
        next = reader.next()
        if next != "END":
            doc_id = next[0]
            raw_term_frequency = next[1]
            dct[doc_id] = raw_term_frequency
        else:
            break
    return dct

################################
# Not used in Final Submission #
################################
def get_document_length(docs_dict, doc):
    # docs_dict in the form { 'US5132543': (2451368, 139), ...}
    # doc_id: (byte_offset, document_length)
    return docs_dict[doc][1]

################################
# Not used in Final Submission #
################################
def get_collection_length(docs_dict):
    # Returns total number of words in the Collection
    # For our purposes, the total number of words = sum of number of words in Title and Abstract ONLY
    collection_length = sum([v[1] for k, v in docs_dict.items()])    
    return collection_length

# ================================
# Vector Space Model querying
# ================================

def vsm_query(query, dictionary, postings_file):
    """
    Processes the free text query and retrieves the document ids of the
    documents containing terms in the query.
    Returns a list of doc_ids in decending order of relevance.
    """
    token_normalized = generate_query_vector(query, dictionary)
    tokens = token_normalized.keys()
    doc_tf_dict = get_document_normalized_term_freq(tokens, dictionary, postings_file)
    return score_documents(token_normalized, doc_tf_dict)

# ================================
# Query Expansion
# ================================

################################
# Not used in Final Submission #
################################
def find_more_relevant_documents(relevant_documents, doc_fields_dict, postings_file):
    """
    Finds more relevant documents based on the given relevant documents' "Family Members"
    and "Cites" fields.
    Returns an expanded list of relevant documents
    """
    all_relevant_docs = set(relevant_documents)
    for doc in relevant_documents:
        if doc in doc_fields_dict:
            reader = get_doc_fields_postings_reader(doc, doc_fields_dict, postings_file)
            while True:
                next_doc = reader.next()
                if next_doc == "END":
                    break
                all_relevant_docs.add(next_doc[0])
    return list(all_relevant_docs)

################################
# Not used in Final Submission #
################################
def generate_query_vector(query, dictionary):
    """
    Returns a dictionary with keys being the tokens present in the query
    and values being the tf-idf values of these tokens.
    """
    # Get all tokens from query and the raw query term frequencies
    tokens = {}
    for token in word_tokenize(query):
        token = stemmer.stem(token).lower()
        if token not in tokens:
            tokens[token] = 0
        tokens[token] += 1

    # Use document frequecies of terms to calculate the tf-idf of tokens
    token_tfidf = {}
    for token in tokens:
        term_freq = tokens[token]
        log_weighted_tf = 1 + log10(term_freq)
        if token in dictionary:
            idf = dictionary[token][1]
            token_tfidf[token] = log_weighted_tf * idf

    # Length normalize the tf-idf values obtained
    normalizer = sqrt(reduce(lambda x, y: x + y**2, token_tfidf.values(), 0))
    token_normalized = {}
    for token in token_tfidf:
        token_normalized[token] = token_tfidf[token] / normalizer
    return token_normalized

# ================================
# Rocchio Algorithm
# ================================

################################
# Not used in Final Submission #
################################
def rocchio_algorithm_query(initial_results, query, term_dict, docs_dict, doc_fields_dict, postings):
    # Make use of Patent's Family and Cites fields to find relevant documents
    relevant_documents = find_more_relevant_documents(initial_results, doc_fields_dict, postings)

    # Find irrelevant docs (documents that are not returned by the query)
    non_relevant_documents = list(set(docs_dict.keys()) - set(relevant_documents))

    # Generate relevant docs vector
    relevant_vector = generate_average_document_vector(relevant_documents, term_dict, docs_dict, postings)

    # Generate non-relevant docs vector
    non_relevant_vector = generate_average_document_vector(non_relevant_documents, term_dict, docs_dict, postings)

    # Generate query vector
    query_vector = generate_query_vector(query, term_dict)

    # Generate document vectors
    doc_tf_dict = get_document_normalized_term_freq(query_vector.keys(), term_dict, postings)

    # Generate rocciho vector
    rocchio_vector = combine_vectors(query_vector, relevant_vector, non_relevant_vector)

    return score_documents(rocchio_vector, doc_tf_dict)

################################
# Not used in Final Submission #
################################
def generate_average_document_vector(doc_ids, term_dict, docs_dict, postings_file):
    """
    Generates a normalized log weighted tf vector for each of the documents provided
    and adds all these vectors together, and divides each component by the total number
    of documents
    """
    total_vector = {}
    num_docs = len(doc_ids)
    for doc_id in doc_ids:
        document_vector = generate_document_vector(doc_id, term_dict, docs_dict, postings_file)
        for token in document_vector:
            if token not in total_vector:
                total_vector[token] = 0
            total_vector[token] += document_vector[token]
    for token in total_vector:
        total_vector[token] /= num_docs
    return total_vector

################################
# Not used in Final Submission #
################################
def combine_vectors(query_vector, relevant_vector, non_relevant_vector):
    """
    Perform Rocchio Algorithm on the three vectors
    Returns an expanded query vector
    """
    query_vector_weight = 0.1
    relevant_vector_weight = 0.2
    non_relevant_vector_weight = -0.7

    vectors = [query_vector, relevant_vector, non_relevant_vector]
    weights = [query_vector_weight, relevant_vector_weight, non_relevant_vector_weight]

    total_vector = {}
    for (i, vector) in enumerate(vectors):
        weight = weights[i]
        for token in vector:
            if token not in total_vector:
                total_vector[token] = 0
            total_vector[token] += weight * vector[token]
    return total_vector

################################
# Not used in Final Submission #
################################
def generate_document_vector(doc_id, term_dict, docs_dict, postings_file):
    """
    Generates a normalized log weighted tf vector for a single document
    """
    log_tf_dict = {}
    reader = get_docs_postings_reader(doc_id, docs_dict, postings_file)
    while True:
        next_token = reader.next()
        if next_token == "END":
            break
        (token, tf) = next_token
        log_tf_dict[token] = (1 + log10(tf)) * term_dict[token][1]
    normalizer = sqrt(reduce(lambda x, y: x + y**2, log_tf_dict.values(), 0))
    normalized_vector = {}
    for token in log_tf_dict:
        normalized_vector[token] = float(log_tf_dict[token]) / float(normalizer)
    return normalized_vector

# ================================
# Vector Space Model Querying
# ================================

def get_document_normalized_term_freq(tokens, dictionary, postings_file):
    """
    Gets the normalized term frequencies for each document containing the
    tokens provided in tokens.
    Since the term frequencies are already log weighted and normalized in
    indexing stage, this function simply retrieves the value using
    PostingReader.
    Returns a dictionary of dictionaries. The outer dictionary is keyed by
    the doc_ids of the documents containing the tokens, and the inner
    dictionary is keyed by the tokens present in the document, with values
    being the normalized term frequencies of that term in that document.
    """
    doc_tf_dict = {}
    for token in tokens:
        if token not in dictionary:
            continue
        reader = get_term_postings_reader(token, dictionary, postings_file)
        next_doc = reader.next()
        while next_doc != "END":
            doc_id = next_doc[0]
            if doc_id not in doc_tf_dict:
                doc_tf_dict[doc_id] = {}
            doc_tf_dict[doc_id][token] = next_doc[2]
            next_doc = reader.next()
    return doc_tf_dict

def score_documents(query_freqs, doc_freqs):
    """
    Scores documents in the provided doc_freqs dictionary based on the values
    in itself and the query_freqs dictionary using cosine similarity.
    Returns a list of doc_ids sorted in the order of highest score to lowest.
    """
    scored_docs = []
    for doc_id in doc_freqs:
        score = 0
        for token in doc_freqs[doc_id]:
            score += doc_freqs[doc_id][token] * query_freqs[token]
        scored_docs.append((doc_id, score))
    # Sort in reverse order

    scored_docs.sort(key=lambda x: (-x[1], x[0]))
    #return map(lambda x: x[0], scored_docs)
    return scored_docs

# =========================================
# Helper functions for PostingsReader
# =========================================

def get_term_postings_reader(token, term_dict, postings_file):
    normalized_tf_parser = lambda x: float("0." + x)
    return PostingReader(postings_file, term_dict[token][0], 3, [str, int, normalized_tf_parser])

def get_docs_postings_reader(doc_id, docs_dict, postings_file):
    return PostingReader(postings_file, docs_dict[doc_id][0], 2, [str, int])

def get_doc_fields_postings_reader(doc_id, doc_fields_dict, postings_file):
    return PostingReader(postings_file, doc_fields_dict[doc_id], 1)

# ===================
# PostingReader
# ===================

class PostingReader:
    """
    PostingReader reads a posting list in a provided postings file object
    using the byte offset provided by a dictionary.
    """
    def __init__(self, postings_file, byte_offset, num_vals, type_converters = None):
        assert type(num_vals) is int and num_vals > 0
        self.postings_file = postings_file
        self.byte_offset = byte_offset
        self.num_vals = num_vals
        if not type_converters or len(type_converters) != num_vals:
            self.type_converters = [lambda x: x for i in range(0, num_vals)]
        else:
            self.type_converters = type_converters
        self.current = 0 # this is the offset that is added to the byte offset when seeking
        self.end = False # set to true when reached end of the list (end of line)
    
    def next(self):
        """
        Gets the next document id and its normalized log weighted term
        frequecy.
        Returns a 2-tuple of the document id and the term frequency.
        If the posting reader has reached the end of the postings list,
        returns 'END'.
        """
        if self.end:
            return "END"
        current_offset = self.current
        result = []
        for i in range(0, self.num_vals):
            val = ""
            while True:
                self.postings_file.seek(self.byte_offset + current_offset)
                next_char = self.postings_file.read(1)
                current_offset += 1
                if next_char == "\n":
                    self.end = True
                    break
                if next_char == " ":
                    break
                val += next_char
            result.append(val)
        self.current = current_offset
        result = map(lambda x: self.type_converters[x](result[x]), range(0, self.num_vals))
        return tuple(result)


def usage():
    print "usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results"

if __name__ == "__main__":
    input_dict_file = input_post_file = input_query_file = output_file = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
    except getopt.GetoptError, err:
        usage()
        sys.exit(2)
    for o, a in opts:
        if o == '-d':
            input_dict_file = a
        elif o == '-p':
            input_post_file = a
        elif o == '-q':
            input_query_file = a
        elif o == '-o':
            output_file = a
        else:
            assert False, "unhandled option"
    if input_dict_file == None or input_post_file == None or input_query_file == None or output_file == None:
        usage()
        sys.exit(2)

    (term_dict, docs_dict, doc_fields_dict, ipc_indices) = build_dict(input_dict_file)
    execute_query(input_post_file, input_query_file, output_file, term_dict, docs_dict, doc_fields_dict, ipc_indices)