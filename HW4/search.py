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

stops = map(lambda x: stemmer.stem(x).lower(), stopwords.words('english'))

def build_dict(input_dict_file):
    """
    Builds the dictionary from the dictionary file. Kept in memory.
    Returns the total number of documents and a dictionary
    """
    mode = -1 # 0: term_dict, 1: doc_dict, 2: fields_dict
    dict_file = file(input_dict_file, 'r')
    term_dict = {}
    docs_dict = {}
    doc_fields_dict = {}
    for line in dict_file:
        line = line.strip()
        if line == "## Term ##":
            mode = 0
        elif line == "## Doc ID ##":
            mode = 1
        elif line == "## Fields ##":
            mode = 2
        elif mode == 0:
            split_line = line.split(" ")
            token = split_line[0]
            byte_offset = int(split_line[1])
            idf = float(split_line[2])
            term_dict[token] = (byte_offset, idf)
        elif mode == 1:
            split_line = line.split(" ")
            doc_id = split_line[0]
            byte_offset = int(split_line[1])
            term_count = int(split_line[2])
            docs_dict[doc_id] = (byte_offset, term_count)
        elif mode == 2:
            split_line = line.split(" ")
            doc_id = split_line[0]
            byte_offset = int(split_line[1])
            doc_fields_dict[doc_id] = byte_offset

    dict_file.close()
    return (term_dict, docs_dict, doc_fields_dict)

def execute_query(input_post_file, input_query_file, output_file, term_dict, docs_dict, doc_fields_dict):
    """
    Tests the queries in the input_query_file based on the dictionary and postings.
    Writes results into output_file.
    """   

    # Initialisation
    tree = ET.parse(input_query_file)
    root = tree.getroot()
    postings = file(input_post_file, 'r')
    output = file(output_file, 'w')

    query = root[0].text.strip() + " " + root[1].text.strip()[33:].strip()

    # Process line
    result = process_query(query.strip(), term_dict, postings)
    output_line = reduce(lambda x, y: x + str(y[0]) + " ", result, "").strip()
    output.write(output_line)

    output.close()

def process_query(query, dictionary, postings_file):
    """
    Processes the free text query and retrieves the document ids of the
    documents containing terms in the query.
    Returns a list of doc_ids in decending order of relevance.
    """
    token_normalized = normalize_query_term_frequencies(query, dictionary)
    doc_tf_dict = get_document_normalized_term_freq(token_normalized.keys(), dictionary, postings_file)
    return score_documents(token_normalized, doc_tf_dict)

def normalize_query_term_frequencies(query, dictionary):
    """
    Returns a dictionary with keys being the tokens present in the query
    and values being the tf-idf values of these tokens.
    """
    # Get all tokens from query and the raw query term frequencies
    tokens = {}
    for token in word_tokenize(query):
        token = stemmer.stem(token).lower()
        if token in stops:
            continue
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
        reader = PostingReader(postings_file, dictionary[token][0])
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

class PostingReader:
    """
    PostingReader reads a posting list in a provided postings file object
    using the byte offset provided by a dictionary.
    """
    def __init__(self, postings_file, byte_offset):
        self.postings_file = postings_file
        self.byte_offset = byte_offset
        self.current = 0 # this is the offset that is added to the byte offset when seeking
        self.end = False # set to true when reached end of the list (end of line)
    def next(self):
        """
        Gets the next document id and it's normalized log weighted term
        frequecy.
        Returns a 2-tuple of the document id and the term frequency.
        If the posting reader has reached the end of the postings list,
        returns 'END'.
        """
        if self.end:
            return "END"
        current_offset = self.current
        doc_id = ""
        while True:
            self.postings_file.seek(self.byte_offset + current_offset)
            next_char = self.postings_file.read(1)
            current_offset += 1
            if next_char == " ":
                break
            doc_id += next_char
        term_freq = ""
        while True:
            self.postings_file.seek(self.byte_offset + current_offset)
            next_char = self.postings_file.read(1)
            current_offset += 1
            if next_char == "\n":
                self.end = True
                break
            if next_char == " ":
                break
            term_freq += next_char
        log_tf = ""
        while True:
            self.postings_file.seek(self.byte_offset + current_offset)
            next_char = self.postings_file.read(1)
            current_offset += 1
            if next_char == "\n":
                self.end = True
                break
            if next_char == " ":
                break
            log_tf += next_char
        self.current = current_offset
        return (doc_id, int(term_freq), float("0." + log_tf))


def usage():
    print "usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results"

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

(term_dict, docs_dict, doc_fields_dict) = build_dict(input_dict_file)
execute_query(input_post_file, input_query_file, output_file, term_dict, docs_dict, doc_fields_dict)