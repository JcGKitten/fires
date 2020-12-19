#!/usr/bin/env python
import sys
import csv
import numpy as np

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    with open(input_file, "r") as f:
        Lines =f.readlines()
        with open(output_file, "w") as o:
            for line in Lines:
                line = line.strip()
                line += "\n"
                o.writelines(line)