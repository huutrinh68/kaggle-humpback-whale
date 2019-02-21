import os
import pandas as pd
import csv

sub_files = [
                 "./input/submission_855.csv",
                 "./input/submission_822.csv",
                 "./input/submission_875.csv",
                 "./input/submission_842.csv",
                 "./input/submission_874.csv",
            ]

sub_weight = [           
                3*0.855**2,
                1*0.822**2,
                5*0.875**2,
                2*0.842**2,
                4*0.874**2,
            ]

Hlabel = 'Image' 
Htarget = 'Id'
npt = 6
place_weights = {}
for i in range(npt):
    place_weights[i] = ( 1 / (i + 1) )
    
print(place_weights)

lg = len(sub_files)
sub = [None]*lg
for i, file in enumerate( sub_files ):
   
    print("Reading {}: w={} - {}". format(i, sub_weight[i], file))
    reader = csv.DictReader(open(file,"r"))
    sub[i] = sorted(reader, key=lambda d: str(d[Hlabel]))

out = open("submission_1.csv", "w", newline='')
writer = csv.writer(out)
writer.writerow([Hlabel,Htarget])

for p, row in enumerate(sub[0]):
    target_weight = {}
    for s in range(lg):
        row1 = sub[s][p]
        for ind, trgt in enumerate(row1[Htarget].split(' ')):
            target_weight[trgt] = target_weight.get(trgt,0) + (place_weights[ind]*sub_weight[s])
    tops_trgt = sorted(target_weight, key=target_weight.get, reverse=True)[:npt]
    writer.writerow([row1[Hlabel], " ".join(tops_trgt)])
out.close()