import json

data = {}

with open("data.json", "r") as f:
    path2arr = json.load(f)

for k in path2arr:
    

with open("data/concept_list.json", "w") as f:
    json.dump(data, f)