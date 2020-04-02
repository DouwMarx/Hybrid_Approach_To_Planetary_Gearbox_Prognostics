import json



#:load the exported json file
with open('dict.json') as json_file:
    dict = json.load(json_file)


    with open("dict.json","w") as outfile:
        json.dump(dict, outfile, indent=8, separators=(',', ': '))



