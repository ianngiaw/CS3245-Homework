#############
# A0121437N #
#############

#!/usr/bin/python
import re
import nltk
import sys
import getopt

# Log function imported from math to be used in probability calculation
from math import log10

# The default ngram size of 4
NGRAM_SIZE = 4

def build_LM(in_file):
    """
    build language models for each label
    each line in in_file contains a label and an URL separated by a tab(\t)
    """
    print 'building language models...'
    # the 4-gram language model
    smoothed_model = {}
    # the total number of occurrences for each language
    totals = {}
    # initialize the languages in the model
    input_file = file(in_file, 'r')
    for line in input_file:
        split_line = line.split(" ", 1)
        language = split_line[0]
        if language not in smoothed_model:
            smoothed_model[language] = {}
            totals[language] = 0
    input_file.close()
    # add data to language models
    input_file = file(in_file, 'r')
    for line in input_file:
        split_line = line.split(" ", 1)
        language = split_line[0]
        # Convert sentence into a list of ngrams
        ngrams = ngrams_for_string(NGRAM_SIZE, split_line[1].strip())
        for ngram in ngrams:
            for lang in smoothed_model:
                if ngram not in smoothed_model[lang]:
                    smoothed_model[lang][ngram] = 1
                    totals[lang] += 1
            smoothed_model[language][ngram] += 1
            totals[language] += 1
    input_file.close()
    return (smoothed_model, totals)
    
def test_LM(in_file, out_file, LM):
    """
    test the language models on new URLs
    each line of in_file contains an URL
    you should print the most probable label for each URL into out_file
    """
    print "testing language models..."
    # This is an empty method
    # Pls implement your code in below
    test_file = file(in_file, 'r')
    output_file = file(out_file, 'w')

    (smoothed_model, totals) = LM

    for line in test_file:
        # Convert sentence into a list of ngrams
        ngrams = ngrams_for_string(NGRAM_SIZE, line.strip())
        # The results of the probability calculation for each language
        results = []
        for language in smoothed_model:
            language_model = smoothed_model[language]
            # The counts of each of the ngrams to be used as numerators in
            # the probability calculation
            numerators = []
            # The number of times a ngram is not found in the language model
            miss_count = 0
            lang_total = float(totals[language])
            for ngram in ngrams:
                if ngram in language_model:
                    numerators.append(language_model[ngram])
                else:
                    miss_count += 1
            # Ignore result if the number of times ngrams of the sentence
            # exceeds 75% of the total number of ngrams being tested for
            # that sentence
            if len(numerators) > 0 and miss_count < len(ngrams) * 0.75:
                probability = 0
                for num in numerators:
                    probability += log10(num / lang_total)
                results.append((language, probability))
        top_lang = 'other'
        if len(results) > 0:
            top_lang = max(results, key = lambda x: x[1])[0]
        output_file.write(top_lang + " " + line)
    test_file.close()
    output_file.close()

def ngrams_for_string(n, string):
    """
    Converts a string into a list of ngrams based on the size provided
    Ngrams are generated WITHOUT start and end padding
    Allows for customization of ngrams for essay questions
    """
    res = []
    char_list = list(string)
    for i in range(0, len(string) - n + 1):
        chars = tuple(char_list[i : i + n])
        res.append(chars)
    return res

########################
# FOR ESSAY QUESTION 1 #
########################
# Test for token based ngram model
# Uncomment the function below to proceed with the test
# def ngrams_for_string(n, sentence):
#     res = []   
#     tokens = sentence.split(" ")
#     import string
#     tokens = filter(lambda x: x not in string.punctuation and x != "", tokens)
#     for i in range(0, len(tokens) - n + 1):
#         ngram = tuple(tokens[i : i + n])
#         res.append(ngram)
#     return res
########################
# END ESSAY QUESTION 1 #
########################

########################
# FOR ESSAY QUESTION 3 #
########################
# Test for strings stripped of numbers and punctuations
# Also tests for strings converted to lowercase
# Uncomment the function below to proceed with the test
# def ngrams_for_string(n, sentence):
#     res = []
#     # Converts all characters to lowercase
#     # sentence = sentence.lower()
#    
#     char_list = list(sentence)
#   
#     # Both punctuation and numbers removed
#     char_list = filter(lambda x: x.isalpha() or x.isspace(), char_list)
#   
#     # Only punctuation removed
#     # import string
#     # char_list = filter(lambda x: x not in string.punctuation, char_list)
#   
#     # Only numeric removed
#     # char_list = filter(lambda x: not x.isdigit(), char_list)
#     for i in range(0, len(char_list) - n + 1):
#         chars = tuple(char_list[i : i + n])
#         res.append(chars)
#     return res
########################
# END ESSAY QUESTION 3 #
########################

########################
# FOR ESSAY QUESTION 4 #
########################
# Tests for other n-gram sizes
# Uncomment the size of the ngram you would like to test
# NGRAM_SIZE = 1
# NGRAM_SIZE = 2
# NGRAM_SIZE = 3
########################
# END ESSAY QUESTION 4 #
########################

def usage():
    print "usage: " + sys.argv[0] + " -b input-file-for-building-LM -t input-file-for-testing-LM -o output-file"

input_file_b = input_file_t = output_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], 'b:t:o:')
except getopt.GetoptError, err:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == '-b':
        input_file_b = a
    elif o == '-t':
        input_file_t = a
    elif o == '-o':
        output_file = a
    else:
        assert False, "unhandled option"
if input_file_b == None or input_file_t == None or output_file == None:
    usage()
    sys.exit(2)

LM = build_LM(input_file_b)
test_LM(input_file_t, output_file, LM)
