#!/usr/bin/python
import re
import nltk
import sys
import getopt

from nltk.tokenize import word_tokenize

stemmer = nltk.stem.porter.PorterStemmer()

from math import log10, sqrt

def build_dict(input_dict_file):
    """
    Builds the dictionary from the dictionary file. Kept in memory.
    Returns the total number of documents and a dictionary
    """
    dict_file = file(input_dict_file, 'r')
    dictionary = {}
    line_count = 0
    total_documents = 0
    for line in dict_file:
        if line_count < 1:
            total_documents = int(line.strip())
            line_count += 1
            continue
        split_line = line.strip().split(" ")
        token = split_line[0]
        byte_offset = int(split_line[1])
        freq = int(split_line[2])
        dictionary[token] = (byte_offset, freq)
    dict_file.close()
    return total_documents, dictionary

def execute_queries(input_post_file, input_query_file, output_file, dictionary, total_documents):
    """
    Tests the queries in the input_query_file based on the dictionary and postings.
    Writes results into output_file.
    """   

    # Initialisation
    queries = file(input_query_file, 'r')   
    postings = file(input_post_file, 'r')
    output = file(output_file, 'w')

    # Reads the query line by line    
    for query in queries.readlines():
        # Process line
        result = process_query(query.strip(), dictionary, postings, total_documents)
        output_line = reduce(lambda x, y: x + str(y) + " ", result[:10], "").strip() + "\n"
        output.write(output_line)

    output.close()

def process_query(query, dictionary, postings_file, total_documents):
    token_dict = {}
    for token in word_tokenize(query):
        token = stemmer.stem(token).lower()
        if token not in token_dict:
            token_dict[token] = 0
        token_dict[token] += 1

    token_tfidf = {}
    for token in token_dict:
        term_freq = token_dict[token]
        log_weighted_tf = 1 + log10(term_freq)
        if token in dictionary:
            doc_freq = dictionary[token][1]
            idf = log10(float(total_documents) / doc_freq)
            token_tfidf[token] = log_weighted_tf * idf

    normalizer = sqrt(reduce(lambda x, y: x + y**2, token_tfidf.values(), 0))
    token_normalized = {}
    for token in token_tfidf:
        if token_tfidf[token] > 0:
            token_normalized[token] = token_tfidf[token] / normalizer

    doc_tf_dict = {}
    for token in token_normalized:
        if token not in dictionary:
            continue
        reader = PostingReader(postings_file, dictionary[token][0])
        next_doc = reader.next()
        while next_doc != "END":
            doc_id = next_doc[0]
            if doc_id not in doc_tf_dict:
                doc_tf_dict[doc_id] = {}
            doc_tf_dict[doc_id][token] = next_doc[1]
            next_doc = reader.next()

    scored_docs = []
    for doc in doc_tf_dict:
        score = 0
        for token in doc_tf_dict[doc]:
            score += doc_tf_dict[doc][token] * token_normalized[token]
        scored_docs.append((doc, score))
    scored_docs.sort(key=lambda x: -x[1])
    return map(lambda x: x[0], scored_docs)

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
        self.current = current_offset
        return (doc_id, float("0." + term_freq))


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

(total_documents, dictionary) = build_dict(input_dict_file)
execute_queries(input_post_file, input_query_file, output_file, dictionary, total_documents)