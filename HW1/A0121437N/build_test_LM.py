#!/usr/bin/python
import re
import nltk
import sys
import getopt
from math import log10

# Code with my model of smoothing
'''
def build_LM(in_file):
    """
    build language models for each label
    each line in in_file contains a label and an URL separated by a tab(\t)
    """
    print 'building language models...'
    # This is an empty method
    # Pls implement your code in below
    smoothed_model = {}
    totals = {}
    input_file = file(in_file, 'r')
    for line in input_file:
        split_line = line.split(" ", 1)
        language = split_line[0]
        if language not in smoothed_model:
            smoothed_model[language] = {}
        if language not in totals:
            totals[language] = 0
        char_list = list(split_line[1].strip())
        for i in range(0, len(char_list) - 3):
            chars = tuple(char_list[i:i+4])
            if chars not in smoothed_model[language]:
                smoothed_model[language][chars] = 2
                totals[language] += 2
            else:
                smoothed_model[language][chars] += 1
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

    smoothed_model = LM[0]
    totals = LM[1]

    for line in test_file:
        char_list = list(line.strip())
        results = []
        for language in smoothed_model:
            current_total = totals[language]
            language_model = smoothed_model[language]
            numerators = []
            miss_count = 0
            for i in range(0, len(char_list) - 3):
                chars = tuple(char_list[i:i+4])
                if chars in language_model:
                    numerators.append(language_model[chars])
                else:
                    numerators.append(1)
                    current_total += 1
                    miss_count += 1
            probability = 0
            for numerator in numerators:
                probability += log10(numerator / float(current_total))
            if miss_count < (len(char_list) - 3) * 0.75:
                results.append((language, probability))
        top_lang = ('other', 0)
        if len(results) > 0:
            top_lang = results[0]
            for result in results[1:]:
                if result[1] > top_lang[1]:
                    top_lang = result
        output_file.write(top_lang[0] + " " + line)
    test_file.close()
    output_file.close()
'''

# Code with defined model of smoothing
def build_LM(in_file):
    """
    build language models for each label
    each line in in_file contains a label and an URL separated by a tab(\t)
    """
    print 'building language models...'
    # This is an empty method
    # Pls implement your code in below

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
        char_list = list(split_line[1].strip())
        for i in range(0, len(char_list) - 3):
            chars = tuple(char_list[i:i+4])
            for lang in smoothed_model:
                if chars not in smoothed_model[lang]:
                    smoothed_model[lang][chars] = 1
                    totals[lang] += 1
            smoothed_model[language][chars] += 1
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

    line_count = 1
    for line in test_file:
        char_list = list(line.strip())
        results = []
        for language in smoothed_model:
            language_model = smoothed_model[language]
            numerators = []
            miss_count = 0
            lang_total = float(totals[language])
            for i in range(0, len(char_list) - 3):
                chars = tuple(char_list[i:i+4])
                if chars in language_model:
                    numerators.append(language_model[chars])
                else:
                    miss_count += 1
            if len(numerators) > 0:
                probability = reduce(lambda x,y: x + log10(y / lang_total), numerators)
                if miss_count < (len(char_list) - 3) * 0.75:
                    results.append((language, probability))
        top_lang = ('other', 0)
        line_count += 1
        if len(results) > 0:
            top_lang = results[0]
            for result in results[1:]:
                if result[1] > top_lang[1]:
                    top_lang = result
        output_file.write(top_lang[0] + " " + line)
    test_file.close()
    output_file.close()

# Unigram test
'''
# Code with defined model of smoothing
def build_LM(in_file):
    """
    build language models for each label
    each line in in_file contains a label and an URL separated by a tab(\t)
    """
    print 'building language models...'
    # This is an empty method
    # Pls implement your code in below

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
        char_list = list(split_line[1].strip())
        for i in range(0, len(char_list)):
            chars = tuple(char_list[i:i+1])
            for lang in smoothed_model:
                if chars not in smoothed_model[lang]:
                    smoothed_model[lang][chars] = 1
                    totals[lang] += 1
            smoothed_model[language][chars] += 1
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

    smoothed_model = LM[0]
    totals = LM[1]

    for line in test_file:
        char_list = list(line.strip())
        results = []
        for language in smoothed_model:
            language_model = smoothed_model[language]
            numerators = []
            miss_count = 0
            for i in range(0, len(char_list)):
                chars = tuple(char_list[i:i+1])
                if chars in language_model:
                    numerators.append(language_model[chars])
                else:
                    miss_count += 1
            new_total = float(totals[language] + miss_count)
            if len(numerators) > 0:
                probability = reduce(lambda x,y: x + log10(y / new_total), numerators)
                if miss_count > 0:
                    probability += reduce(lambda x,y: x + log10(1.0 / new_total), range(0, miss_count))
                if miss_count < (len(char_list) - 3) * 0.75:
                    results.append((language, probability))
        top_lang = ('other', 0)
        if len(results) > 0:
            top_lang = results[0]
            for result in results[1:]:
                if result[1] > top_lang[1]:
                    top_lang = result
        output_file.write(top_lang[0] + " " + line)
    test_file.close()
    output_file.close()
'''

# Bigram model
'''
# Code with defined model of smoothing
def build_LM(in_file):
    """
    build language models for each label
    each line in in_file contains a label and an URL separated by a tab(\t)
    """
    print 'building language models...'
    # This is an empty method
    # Pls implement your code in below

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
        char_list = list(split_line[1].strip())
        for i in range(0, len(char_list)-1):
            chars = tuple(char_list[i:i+2])
            for lang in smoothed_model:
                if chars not in smoothed_model[lang]:
                    smoothed_model[lang][chars] = 1
                    totals[lang] += 1
            smoothed_model[language][chars] += 1
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

    smoothed_model = LM[0]
    totals = LM[1]

    for line in test_file:
        char_list = list(line.strip())
        results = []
        for language in smoothed_model:
            language_model = smoothed_model[language]
            numerators = []
            miss_count = 0
            for i in range(0, len(char_list)-1):
                chars = tuple(char_list[i:i+2])
                if chars in language_model:
                    numerators.append(language_model[chars])
                else:
                    miss_count += 1
            new_total = float(totals[language] + miss_count)
            if len(numerators) > 0:
                probability = reduce(lambda x,y: x + log10(y / new_total), numerators)
                if miss_count > 0:
                    probability += reduce(lambda x,y: x + log10(1.0 / new_total), range(0, miss_count))
                if miss_count < (len(char_list) - 3) * 0.75:
                    results.append((language, probability))
        top_lang = ('other', 0)
        if len(results) > 0:
            top_lang = results[0]
            for result in results[1:]:
                if result[1] > top_lang[1]:
                    top_lang = result
        output_file.write(top_lang[0] + " " + line)
    test_file.close()
    output_file.close()
'''

# Trigram model
'''
# Code with defined model of smoothing
def build_LM(in_file):
    """
    build language models for each label
    each line in in_file contains a label and an URL separated by a tab(\t)
    """
    print 'building language models...'
    # This is an empty method
    # Pls implement your code in below

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
        char_list = list(split_line[1].strip())
        for i in range(0, len(char_list)-2):
            chars = tuple(char_list[i:i+3])
            for lang in smoothed_model:
                if chars not in smoothed_model[lang]:
                    smoothed_model[lang][chars] = 1
                    totals[lang] += 1
            smoothed_model[language][chars] += 1
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

    smoothed_model = LM[0]
    totals = LM[1]

    for line in test_file:
        char_list = list(line.strip())
        results = []
        for language in smoothed_model:
            language_model = smoothed_model[language]
            numerators = []
            miss_count = 0
            for i in range(0, len(char_list)-2):
                chars = tuple(char_list[i:i+3])
                if chars in language_model:
                    numerators.append(language_model[chars])
                else:
                    miss_count += 1
            new_total = float(totals[language] + miss_count)
            if len(numerators) > 0:
                probability = reduce(lambda x,y: x + log10(y / new_total), numerators)
                if miss_count > 0:
                    probability += reduce(lambda x,y: x + log10(1.0 / new_total), range(0, miss_count))
                if miss_count < (len(char_list) - 3) * 0.75:
                    results.append((language, probability))
        top_lang = ('other', 0)
        if len(results) > 0:
            top_lang = results[0]
            for result in results[1:]:
                if result[1] > top_lang[1]:
                    top_lang = result
        output_file.write(top_lang[0] + " " + line)
    test_file.close()
    output_file.close()
'''

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
