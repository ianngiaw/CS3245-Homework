#!/usr/bin/python
import re
import nltk
import sys
import getopt

# For testing convenience, remove when submitting
# python search.py -d dict.txt -p postings.txt -q queries.txt -o output.txt

def build_dict(input_dict_file):
    """
    Builds the dictionary from the dictionary file. Kept in memory.
    Returns a dictionary
    """
    dict_file = file(input_dict_file, 'r')
    dictionary = {}
    for line in dict_file:
        split_line = line.strip().split(" ")
        token = split_line[0]
        byte_offset = int(split_line[1])
        freq = int(split_line[2])
        dictionary[token] = (byte_offset, freq)
    dict_file.close()
    return dictionary

def execute_queries(input_post_file, input_query_file, output_file, dictionary):
    """
    Tests the queries in the input_query_file based on the dictionary and postings.
    Writes results into output_file.
    """   

    # Initialisation
    queries = file(input_query_file, 'r')   
    postings = file(input_post_file, 'r') 
    output_id = "" # String containing all the required doc ids

    # Reads the query line by line    
    for query in queries.readlines():
        # Parse each query
        print "Query: " + query.strip()
        
        # Shunting-Yard algorithm
        output_queue = []
        operator_stack = []
        operators = {"OR": 1, "AND": 2, "NOT": 3 , "(": 4, ")": 4}

        for word in query.strip().split():                
            # Token is an Operator
            if word in operators:
                # Parenthesis checks
                if word == "(":
                    operator_stack.append(word)
                elif word == ")":
                    # need to check the whole stack until a "(" is found (troublesome)
                    while len(operator_stack) > 0:
                        if operator_stack[-1] != "(":
                            output_queue.append(operator_stack.pop())
                        else:
                            operator_stack.pop()
                    if len(operator_stack) > 0 and operator_stack[-1] != "(":
                        output_queue.append(operator_stack.pop())
                else:
                    # Push onto stack if stack is empty
                    if len(operator_stack) == 0:
                        operator_stack.append(word)
                    else:
                        while len(operator_stack) > 0 and operators[operator_stack[-1]] > operators[word]:
                            # Pop the operator from the stack and add it to output
                            output_queue.append(operator_stack.pop())
                        operator_stack.append(word)

            # Token is a Word
            else:
                output_queue.append(word)

        # Empty out the operator stack into the output queue
        while len(operator_stack) > 0:
            output_queue.append(operator_stack.pop())

        # Reverse Polish Notation debug
        print output_queue

    return dictionary

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
        Retrieves the next doc id in the postings list
        """
        if self.end:
            return "END"
        current_offset = self.current
        self.postings_file.seek(self.byte_offset + current_offset)
        parsed_string = self.postings_file.read(1)
        current_offset += 1
        
        # Encounters a skip pointer, denoted in our postings file by a '*'
        is_skip = parsed_string == "*"
        if is_skip:
            # "*" in the postings list file indicates the number after it is a skip pointer
            parsed_string = "" 

        while True:
            self.postings_file.seek(self.byte_offset + current_offset)
            next_char = self.postings_file.read(1)
            if next_char == " ":
                current_offset += 1
                break
            if next_char == "\n":
                # End of line reached
                self.end = True
                break
            parsed_string += next_char
            current_offset += 1
        self.current = current_offset
        
        if is_skip:
            # Returns a 3-tuple, the last being the new current if the skip pointer is used
            skip_gap = int(parsed_string)
            return (True, self.get_skip_value(skip_gap), self.current + skip_gap)

        return (False, int(parsed_string))
    
    def get_skip_value(self, skip_gap):
        parsed_string = ""
        while True:
            self.postings_file.seek(self.byte_offset + self.current + skip_gap)
            next_char = self.postings_file.read(1)
            if next_char == " " or next_char == "\n":
                break
            parsed_string += next_char
            skip_gap += 1
        return int(parsed_string)
    
    def skip_to(self, new_current):
        """
        Sets the current to the provided new_current value
        """
        self.current = new_current

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
execute_queries(input_post_file, input_query_file, output_file, dictionary)