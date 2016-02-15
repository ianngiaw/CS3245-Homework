#!/usr/bin/python
import re
import nltk
import sys
import getopt

def build_dict(input_dict_file):
    """
    Builds the dictionary from the dictionary file.
    Returns a dictionary
    """
    return

def text_index(input_post_file, input_query_file, output_file, dictionary):
    """
    Tests the queries in the input_query_file based on the dictionary and postings.
    Writes results into output_file.
    """
    return


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

dictionary = build_dict(input_dict_file)
test_index(input_post_file, input_query_file, output_file, dictionary)