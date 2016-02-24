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
        
        # Construct Reverse Polish Notation
        rpn_lst = shunting_yard(query)
        print rpn_lst

        print rpn_interpreter(rpn_lst, postings)
        # print dictionary

    return

def and_query(t1, t2, postings):
    # Need to handle strings and list(s) of doc ids differently
    output = []

    if type(t1) ==  str and type(t2) == str:
        # Do the merge algorithm as discussed in lectures
        t1_offset = dictionary[t1][0]
        t2_offset = dictionary[t2][0]

        t1_reader = PostingReader(postings, t1_offset)
        t2_reader = PostingReader(postings, t2_offset)
        
        print t1_reader.next()
        print t1_reader.next()
        print t1_reader.next()
        print t1_reader.next()
        print t1_reader.next()
        print t1_reader.next()
        print t1_reader.next()
        print t1_reader.next()
        print t1_reader.next()
        print t1_reader.next()
        print t1_reader.next()
        print t1_reader.next()

        output += ["hello",]

        return output
    elif (type(t1) == str and type(t2) == list) or (type(t1) == list and type(t2) == str):
        # One of them is a str, the other is a list
        # Need to transform the str into a list of doc ids, then do merging
        pass
    else:
        # Both of them are lists, easiest to merge
        t1_ptr = 0
        t2_ptr = 0
        while t1_ptr < len(t1) and t2_ptr < len(t2):
            if t1[t1_ptr] == t2[t2_ptr]:
                output.append(t1[t1_ptr])
                t1_ptr += 1
                t2_ptr += 1
            elif t1[t1_ptr] < t2[t2_ptr]:
                t1_ptr += 1
            else:
                t2_ptr += 1

    return output

def or_query(t1, t2, postings):
    # Need to handle strings and list(s) of doc ids differently
    output = []

    if type(t1) ==  str and type(t2) == str:
        # Do the merge algorithm as discussed in lectures
        pass
    elif (type(t1) == str and type(t2) == list) or (type(t1) == list and type(t2) == str):
        # Either one of them is a str, the other is a list
        # Need to transform the str into a list of doc ids, then do merging
        pass
    else:
        # Both of them are lists, easiest to merge
        output = t1 + t2
        output.sort()

    return output

def not_query(t, postings):
    if type(t) ==  str:
        # Transform string to list of doc ids, then subtract from all doc ids
        pass
    else:
        # Filter out from the list of all doc ids 
        pass

# RPN interpreter
def rpn_interpreter(rpn_lst, postings):
    # Initialisation
    binary_operators = {"OR", "AND"}
    operators = set.union(binary_operators, {"NOT"})
    stack = []

    while len(rpn_lst) > 0:
        token = rpn_lst.pop(0) # first item in the list
        if token not in operators:
            # Transform word tokens into a list of doc ids
            # Actually shouldn't! If we do that we're wasting the skip pointers!!
            stack.append(token)
        else:
            if token in binary_operators:
                t1 = stack.pop()
                t2 = stack.pop()
                if token == "OR":
                    stack.append(or_query(t1, t2, postings))
                else:
                    stack.append(and_query(t1, t2, postings))
            else:
                t = stack.pop()
                stack.append(not_query(t, postings))
    
    output = stack.pop() # either a string, or a list of docIds
    if type(output) == str:
        # Find all occurrences of output in the dictionary
        pass
    else:
        return output   

# Shunting-Yard algorithm
def shunting_yard(query_line):
    output_queue = []
    operator_stack = []
    operators = {"OR": 1, "AND": 2, "NOT": 3 , "(": 0, ")": 0}

    for word in nltk.tokenize.word_tokenize(query_line):                
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
                        break
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
    # print output_queue

    return output_queue

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