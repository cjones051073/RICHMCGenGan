#!/usr/bin/env python 

import gzip, bz2

#f = open('data.txt','r')
f = gzip.open('data/PID-train-data.txt.gz')
#f = bz2.BZ2File('data.txt.bz2')

data = f.readlines()

iData   = 0
maxData = 5

for d in data :

    a = d.split()

    print( a )
    
    iData += 1
    if iData >= maxData : break
