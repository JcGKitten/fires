#!/usr/bin/env python
import sys
import csv
import random



if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    with open(input_file, "r") as f:
        Lines = f.readlines()

    head = Lines[0]
    Lines = Lines[1:]
    random.shuffle(Lines)
    with open(output_file, 'w') as w:
        w.writelines(head)
        w.writelines(Lines)
        