#!/usr/bin/python
import search

(term_dict, docs_dict, doc_fields_dict) = search.build_dict("dictionary.txt")

postings = file("postings.txt")

reader1 = search.get_term_postings_reader("determin", term_dict, postings)
reader2 = search.get_docs_postings_reader("US20090112905A1", docs_dict, postings)
reader3 = search.get_doc_fields_postings_reader("US20090289046A1", doc_fields_dict, postings)

readers = [reader1, reader2, reader3]

for i in range(0, len(readers)):
	print "Reader", (i+1)
	reader = readers[i]
	next = None
	while True:
		next = reader.next()
		if next != "END":
			print next
		else:
			break