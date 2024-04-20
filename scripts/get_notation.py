#!/bin/python3

puiss = -45 # dBm
res= ""
if puiss > -85:
    res = "Bonne"
elif puiss < -86 and puiss > -105:
    res = "Moyenne"
elif puiss < -106 and  puiss > -141:
    res = "Mauvaise"
