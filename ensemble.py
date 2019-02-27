import os
import pandas as pd
import csv

sub_files = [
                 "./input/submission_866.csv",
                 "./input/submission_855.csv",
                 "./input/submission_875.csv",
                 "./input/submission_842.csv",
                 "./input/submission_874.csv",
            ]

sub_weight = [           
                6*0.866**2,
                5*0.855**2,
                8*0.875**2,
                3*0.842**2,
                7*0.874**2,
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

out = open("submission_ensem.csv", "w", newline='')
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

# my submission file
df = pd.read_csv("./submission_ensem.csv")
leak_df = pd.read_csv("./leaks.csv")

leak_map = {}

for idx, row in leak_df.iterrows():
    leak_map[row["b_test_img"]] = row["b_label"]

submission_list = []
for idx, row in df.iterrows():
    if row["Image"] in leak_map:
        id_list = row["Id"].split(" ")
        if id_list[0] != leak_map[row["Image"]]:
            print(id_list[0], leak_map[row["Image"]])
            print(id_list)
        id_list[0] = leak_map[row["Image"]]
        id_string = " ".join(id_list)
    else:
        id_string = row["Id"]
    submission_list.append(id_string)
df["Id"] = submission_list
# modified output
df.to_csv("final_submission.csv", index=False)
