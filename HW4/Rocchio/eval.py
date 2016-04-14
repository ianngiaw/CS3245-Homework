#!/usr/bin/python
import subprocess

# Call index first if needed
# subprocess.call(["python","index.py","-i","./patsnap-corpus/","-d","dictionary.txt","-p","postings.txt"])
# subprocess.call(["python","search.py","-d","dictionary.txt","-p","postings.txt","-q","q1.xml","-o","output1.txt"])
# subprocess.call(["python","search.py","-d","dictionary.txt","-p","postings.txt","-q","q2.xml","-o","output2.txt"])

def print_list_rankings(lst, output_list):
	for (i, doc) in enumerate(lst):
		if doc in output_list:
			print i, doc, output_list.index(doc)
		else:
			print i, doc, "not found"

def calculate_average_f2(output_list, positive_list, negative_list):
	positive_set = set(positive_list)
	negative_set = set(negative_list)
	relevant_total = float(len(positive_set))
	relevant_count = 0
	non_relevant_count = 0
	total_f2 = 0
	count = 0
	for doc in output_list:
		if doc in positive_set:
			relevant_count += 1
		elif doc in negative_set:
			non_relevant_count += 1
		precision = 0
		if relevant_count + non_relevant_count > 0:
			precision = float(relevant_count) / float(relevant_count + non_relevant_count)
		recall = float(relevant_count) / relevant_total
		if doc in positive_set:
			total_f2 += calculate_f2(precision, recall)
			count += 1
	return total_f2 / count


def calculate_f2(precision, recall):
	if precision + recall > 0:
		return 5 * precision * recall / (4 * precision + recall)
	return 0

results = []

for i in range(0, 4):
	for r in range(0, 4):
		n = -(10 - i - r)
		for t in range(10, 21, 2):
			for v in range(1, 8, 2):
				v = float(v) / float(10)
				subprocess.call(["python","search.py","-d","dictionary.txt","-p","postings.txt","-q","q1.xml","-o","output1.txt","-i",str(i),"-r",str(r),"-n",str(n),"-t",str(t),"-v",str(v)])
				subprocess.call(["python","search.py","-d","dictionary.txt","-p","postings.txt","-q","q2.xml","-o","output2.txt","-i",str(i),"-r",str(r),"-n",str(n),"-t",str(t),"-v",str(v)])

				# Evaluate query 1
				q1_positive_file = file('q1-qrels+ve.txt')
				q1_negative_file = file('q1-qrels-ve.txt')

				q1_pos_list = list(set([l.strip() for l in q1_positive_file]))
				q1_neg_list = list(set([l.strip() for l in q1_negative_file]))

				q1_output_file = file('output1.txt')
				q1_output_list = q1_output_file.readline().strip().split(" ")

				# Evaluate query 2
				q2_positive_file = file('q2-qrels+ve.txt')
				q2_negative_file = file('q2-qrels-ve.txt')

				q2_pos_list = list(set([l.strip() for l in q2_positive_file]))
				q2_neg_list = list(set([l.strip() for l in q2_negative_file]))

				q2_output_file = file('output2.txt')
				q2_output_list = q2_output_file.readline().strip().split(" ")

				q1_average_f2 = calculate_average_f2(q1_output_list, q1_pos_list, q1_neg_list)
				q2_average_f2 = calculate_average_f2(q2_output_list, q2_pos_list, q2_neg_list)
				total_average_f2 = (q1_average_f2 + q2_average_f2) / 2

				results.append((total_average_f2, q1_average_f2, q2_average_f2, i, r, n, t, v))

				print "###################"
				print "i", i, "r", r, "n", n, "t", t, "v", v
				print "q1's average f2:", q1_average_f2
				print "q2's average f2:", q2_average_f2
				print "total average f2:", total_average_f2
				print "best avg so far:", max(results, key=lambda x: x[0])
				print "best q1 so far:", max(results, key=lambda x: x[1])
				print "best q2 so far:", max(results, key=lambda x: x[2])
				print "###################"

results.sort(key=lambda x: -x[0])
print "TOP AVERAGE"
count = 1
for (a, q1f2, q2f2, i, r, n, t, v) in results[:30]:
	print count, a, q1f2, q2f2, i, r, n, t, v
	count += 1

results.sort(key=lambda x: -x[1])
print "TOP Q1"
count = 1
for (a, q1f2, q2f2, i, r, n, t, v) in results[:30]:
	print count, a, q1f2, q2f2, i, r, n, t, v
	count += 1

results.sort(key=lambda x: -x[2])
print "TOP Q2"
count = 1
for (a, q1f2, q2f2, i, r, n, t, v) in results[:30]:
	print count, a, q1f2, q2f2, i, r, n, t, v
	count += 1