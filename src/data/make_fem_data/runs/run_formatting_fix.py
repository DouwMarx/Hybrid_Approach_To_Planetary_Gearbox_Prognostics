import json



#:load the exported json file
with open('run_template.json') as json_file:
    dict = json.load(json_file)


    with open("run_template.json", "w") as outfile:
        json.dump(dict, outfile, indent=8, separators=(',', ': '))



