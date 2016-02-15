#!/usr/bin/python
import re
import nltk
import sys
import getopt

from os import listdir
from os.path import isfile, join

from nltk.tokenize import sent_tokenize, word_tokenize

stemmer = nltk.stem.porter.PorterStemmer()

def build_index(document_dir):
    """
    Builds the index
    """
    index = {}
    files = listdir(document_dir)
    files.remove(".DS_Store")
    files.sort(key=lambda f: int(f))
    for f in files:
        path = join(document_dir, f)
        if isfile(path):
            input_file = file(path, 'r')
            for line in input_file:
                for sent in sent_tokenize(line):
                    for word in word_tokenize(sent):
                        stemmed_word = stemmer.stem(word)
                        token = stemmed_word.lower()
                        if token not in index:
                            index[token] = []
                        if len(index[token]) == 0 or index[token][-1] != f:
                            index[token].append(f)
    return index

def write_index(output_dict_file, output_post_file, index):
    """
    Writes the index
    """
    return

def usage():
    print "usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file"

document_dir = output_dict_file = output_post_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError, err:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == '-i':
        document_dir = a
    elif o == '-d':
        output_dict_file = a
    elif o == '-p':
        output_post_file = a
    else:
        assert False, "unhandled option"
if document_dir == None or output_dict_file == None or output_post_file == None:
    usage()
    sys.exit(2)

index = build_index(document_dir)
write_index(output_dict_file, output_post_file, index)