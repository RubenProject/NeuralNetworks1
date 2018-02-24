import csv



with open("test_in.csv", 'rb') as f:
	reader = csv.reader(f)
	test_in = list(reader)


with open("test_out.csv", 'rb') as f:
	reader = csv.reader(f)
	test_out = list(reader)


