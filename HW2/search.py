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
    output = file(output_file, 'w')

    # Reads the query line by line    
    for query in queries.readlines():
        # Construct Reverse Polish Notation
        rpn_lst = shunting_yard(query)
        result = rpn_interpreter(dictionary, rpn_lst, postings)
        output_line = reduce(lambda x, y: x + str(y) + " ", result, "").strip() + "\n"
        output.write(output_line)

def and_query(t1_reader, t2_reader):
    # Need to handle strings and list(s) of doc ids differently
    output = []

    while t1_reader.peek() != "END" and t2_reader.peek() != "END":
        t1_id = t1_reader.peek()
        t2_id = t2_reader.peek()

        if t1_id[0] and t2_id[0]:
            t1_reader.next()
            t2_reader.next()
        elif t1_id[0] and not t2_id[0]:
            if t1_id[1] <= t2_id[1]:
                t1_reader.skip_to(t1_id[2])
            else:
                t1_reader.next()
        elif t2_id[0] and not t1_id[0]:
            if t2_id[1] <= t1_id[1]:
                t2_reader.skip_to(t2_id[2])
            else:
                t2_reader.next()
        elif t1_id[1] < t2_id[1]:
                t1_reader.next()
        elif t1_id[1] > t2_id[1]:
            t2_reader.next()
        else:
            output.append(t1_id[1])
            t1_reader.next()
            t2_reader.next()

    return output

def or_query(t1_reader, t2_reader):
    # Need to handle strings and list(s) of doc ids differently
    output = []

    while t1_reader.peek() != "END" and t2_reader.peek() != "END":
        t1_id = t1_reader.peek()
        t2_id = t2_reader.peek()

        # Ignore all skip pointers
        if t1_id[0]:
            t1_reader.next()
            t1_id = t1_reader.peek()
        if t2_id[0]:
            t2_reader.next()
            t2_id = t2_reader.peek()            

        if t1_id[1] == t2_id[1]:
            output += [t1_id[1]]
            t1_reader.next()
            t2_reader.next()
        elif t1_id[1] < t2_id[1]:
            output += [t1_id[1]]
            t1_reader.next()
        else:
            output += [t2_id[1]]
            t2_reader.next()

    return output

def not_query(t_reader):
    output = []
    all_reader = PostingReader(postings, 0)

    while t_reader.peek() != "END":
        t_id = t_reader.peek()
        curr_id = all_reader.peek()

        if t_id[0]:
            t_reader.next()
            t_id = t_reader.peek() 
        if curr_id[0]:
            all_reader.next()
            curr_id = all_reader.peek()            
        
        if curr_id[1] == t_id[1]:
            all_reader.next()
            t_reader.next()
        elif curr_id[1] < t_id[1]:
            output += [curr_id[1]]
            all_reader.next()

    # Add remaining of all postings into the output
    while all_reader.peek() != "END":
        curr_id = all_reader.peek()
        if curr_id[0] == True:
            all_reader.next()
            curr_id = all_reader.peek()
        output += [curr_id[1]]
        all_reader.next()
    
    return output

# RPN interpreter
def rpn_interpreter(dictionary, rpn_lst, postings):
    # Initialisation
    binary_operators = {"OR", "AND"}
    operators = set.union(binary_operators, {"NOT"})
    stack = []

    while len(rpn_lst) > 0:
        token = rpn_lst.pop(0) # first item in the list
        if token not in operators:
            # Transform word tokens into a list of doc ids
            # Actually shouldn't! If we do that we're wasting the skip pointers!!
            if token in dictionary:
                stack.append(PostingReader(postings, dictionary[token][0]))
            else:
                stack.append(MergedPostingReader([]))
            # stack.append(token)
        else:
            query_result = []
            if token in binary_operators:
                t1 = stack.pop()
                t2 = stack.pop()
                if token == "OR":
                    query_result = or_query(t1, t2)
                else:
                    # print "appending", and_query(t1, t2, postings)
                    query_result = and_query(t1, t2)
            else:
                # token is unary operator: NOT
                t = stack.pop()
                query_result = not_query(t)
            stack.append(MergedPostingReader(query_result))
    
    return stack.pop().to_list()

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

class MergedPostingReader:
    """
    MergedPostingReader reads a python list object and returns
    it in the same format as PostingReader.
    """
    def __init__(self, merged_list):
        self.merged_list = merged_list
        self.current = 0

    def peek(self):
        if self.current >= len(self.merged_list):
            return "END"
        return (False, self.merged_list[self.current])

    def next(self):
        self.current += 1
        if self.current >= len(self.merged_list):
            return "END"
        return (False, self.merged_list[self.current])

    def to_list(self):
        return self.merged_list   


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

    def peek(self):
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
                break
            parsed_string += next_char
            current_offset += 1

        if is_skip:
            # Returns a 3-tuple, the last being the new current if the skip pointer is used
            skip_gap = int(parsed_string)
            return (True, self.get_skip_value(current_offset, skip_gap), current_offset + skip_gap)

        return (False, int(parsed_string))
    
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
            return (True, self.get_skip_value(current_offset, skip_gap), current_offset + skip_gap)

        return (False, int(parsed_string))
    
    def get_skip_value(self, current_offset, skip_gap):
        parsed_string = ""
        while True:
            self.postings_file.seek(self.byte_offset + current_offset + skip_gap)
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

    def to_list(self):
        temp_current = self.current
        temp_end = self.end

        result = []
        self.current = 0
        self.end = False

        while self.peek() != "END":
            next_item = self.peek()
            if not next_item[0]:
                result.append(next_item[1])
            self.next()

        self.current = temp_current
        self.send = temp_end
        return result

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