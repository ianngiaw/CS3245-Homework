#!/usr/bin/python
import subprocess

# Call index first if needed
# subprocess.call(["python","index.py","-i","./patsnap-corpus/","-d","dictionary.txt","-p","postings.txt"])
subprocess.call(["python","search.py","-d","dictionary.txt","-p","postings.txt","-q","q1.xml","-o","output1.txt"])
subprocess.call(["python","search.py","-d","dictionary.txt","-p","postings.txt","-q","q2.xml","-o","output2.txt"])

def print_list_rankings(lst, output_list):
	for (i, doc) in enumerate(lst):
		if doc in output_list:
			print i, doc, output_list.index(doc)
		else:
			print i, doc, "not found"

# Evaluate query 1
q1_positive_file = file('q1-qrels+ve.txt')
q1_negative_file = file('q1-qrels-ve.txt')

q1_pos_list = list(set([l.strip() for l in q1_positive_file]))
q1_neg_list = list(set([l.strip() for l in q1_negative_file]))

q1_output_file = file('output1.txt')
q1_output_list = q1_output_file.readline().strip().split(" ")

print "q1 +ve"
print_list_rankings(q1_pos_list, q1_output_list)
print "q1 -ve"
print_list_rankings(q1_neg_list, q1_output_list)

# Evaluate query 2
q2_positive_file = file('q2-qrels+ve.txt')
q2_negative_file = file('q2-qrels-ve.txt')

q2_pos_list = list(set([l.strip() for l in q2_positive_file]))
q2_neg_list = list(set([l.strip() for l in q2_negative_file]))

q2_output_file = file('output2.txt')
q2_output_list = q2_output_file.readline().strip().split(" ")

print "q2 +ve"
print_list_rankings(q2_pos_list, q2_output_list)
print "q2 -ve"
print_list_rankings(q2_neg_list, q2_output_list)