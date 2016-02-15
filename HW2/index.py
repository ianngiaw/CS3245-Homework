#!/usr/bin/python
import re
import nltk
import sys
import getopt

def build_index(document_dir):
    """
    Builds the index
    """
    return

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