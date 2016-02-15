#!/usr/bin/python
import re
import nltk
import sys
import getopt

from os import listdir
from os.path import isfile, join

from nltk.tokenize import sent_tokenize, word_tokenize

from math import sqrt

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
    Writes the index to the output dictionary file and postings file
    """
    dict_file = file(output_dict_file, "w")
    post_file = file(output_post_file, "w")
    count_bytes = 0
    for token in index:
        postings = index[token]
        postings_string = generate_postings_string(postings)
        dict_string = token + " " + str(count_bytes) + " " + str(len(postings)) + "\n"
        dict_file.write(dict_string)
        post_file.write(postings_string)
        count_bytes += len(postings_string)
    dict_file.close()
    post_file.close()

def generate_postings_string(postings):
    """
    Generates a string that is written to a postings file that marks skip pointers with a * character.
    The integer of the skip pointer is the number of bytes after the space after the skip pointer to
    the doc id that it is pointing to.
    For example, if the following postings list is passed ['1', '2', '3', '4', '5']
    The output string is "1 *2 2 3 *2 4 5".
    1 is the first doc id, it has a skip pointer which points to 2 bytes after the space after the
    skip pointer. That is, "1 *2 ^2 3 *2 4 5", 2 bytes after the ^, which points to 3.
    """
    skip_gap = int(sqrt(len(postings)))
    count = 0
    string = ""
    for doc_id in postings:
        string += doc_id + " "
        count += 1
        if skip_gap != 1 and count % skip_gap == 1 and count + skip_gap <= len(postings):
            byte_gap = len(reduce(lambda x, y: x + y + " ", postings[count:count + skip_gap - 1], ""))
            string += "*" + str(byte_gap) + " "
    return string.strip() + "\n"

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