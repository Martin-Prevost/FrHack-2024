#!/bin/python3
import re

filename = "../Fichiers pour challenge 3/Mesures sur 41 45 89.csv"

operateurs = {
    "OP1": [],
    "OP2": [],
    "OP3": [],
    "OP4": []
}

c = 0

with open(filename,"r") as file:
    for line in file:
        if c != 0:
            csv_data = line.split(';')
            csv_data[-1].replace('\n','')
            OP = re.sub(r'_.*', '', csv_data[0])
            operateurs[OP].append(line)
        c += 1


