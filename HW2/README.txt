This is the README file for A01220811X-A0121437N's submission

== General Notes about this assignment ==

Place your comments or requests here for Min to read.  Discuss your
architecture or experiments in general.  A paragraph or two is usually
sufficient.

index.py contains the functions used to build the index. These functions are build_index, write_index and generate_postings_string. The build_index function reads each indivual file in the corpus and splits them into word tokens using the nltk word_tokenizer method. This function returns a python dictionary with keys being the word tokens and the values being the list of document ids of the occurrences of the tokens. This dictionary is then passed to the write_index function which iterates over they keys of the dictionary and generates a 'postings string', written to the postings output file.

To generate the string for a postings list, the generate_postings_string function is called which converts an ordered list of document ids into a string, with skip pointers included. The following example explains the format of the output string.

Assume the postings list provided is [1, 2, 3, 4, 5]. The generate_postings_string function will then return '1 *2 2 3 *2 4 5'. The '*' character indicates that the number following it is a skip pointer. This skip pointer is the number of bytes after the space (' ') character after the skip pointer. So the first '*2' points to 2 bytes after the space after the '*2', which is the '3' character.

search.py contains the logic to parse the Boolean queries. It uses a modified form of the Shunting-Yard algorithm and transforms the Boolean query into a list of tokens that will be evaluated by the rpn_interpreter function. For example, "martha AND monthli" is transformed into ['martha', 'monthli', 'AND'] by the function shunting_yard. rpn_interpreter then takes in the list and executes the necessary functions. 

When word tokens are encountered in the list, a PostingReader object is instantiated with the word and pushed into the a stack. Whenever an operator is encountered, items from the stack is popped and passed into the appropriate function. Every operator returns a python list of document ids, which are then instantiated with a MergedPostingReader and pushed back into the stack. PostingReader and MergedPostingReader provide next() and peek() functions which provides a layer of abstraction over the file seek() and read() functions.

== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.

index.py: Creates the dictionary and postings from the supplied corpus
search.py: Takes in a query file and writes the output to a specified output file
dictionary.txt: The dictionary file index.py produced when run on our machines
postings.txt: The postings file index.py produced when run on our machines

== Statement of individual work ==

Please initial one of the following statements.

[X] I, A0122081X-A0121437N, certify that I have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I
expressly vow that I have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[ ] I, A0122081X-A0121437N, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

I suggest that I should be graded as follows:

<Please fill in>

== References ==

<Please list any websites and/or people you consulted with for this
assignment and state their role>

https://en.wikipedia.org/wiki/Shunting_yard_algorithm: for the Shunting Yard algorithm
https://en.wikipedia.org/wiki/Reverse_Polish_notation: for the algorithm to interpret the output returned from Shunting Yard algorithm
http://stackoverflow.com/questions/3207219/how-to-list-all-files-of-a-directory-in-python: for listing files in directory
