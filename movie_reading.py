import json
import csv
import pandas as pd


file = 'moviequotes.csv'

df = pd.read_csv(file, header='infer')
print("the corpus is ", df.shape)


qs = []
ans = []
tag = []
cntxt = []
a = 0

for i in range(df.shape[0]):
    diggy = df.iloc[i, 0]
    diggy = str(diggy)
    if diggy[0].isdigit():

        res = ''.join([i for i in diggy if not i.isdigit()])

        if a == 0:
            x = res.split(" ")
            taggy = x[1]
            tag.append(taggy)
            qs.append(res)
            a = 1
        elif a == 1:
            ans.append(res)
            a = 0
            cntxt.append("")

zs = qs
bob = len(qs) - 1
qs = qs[0:bob]
tag = tag[0:bob]
print("the length of qs is ", len(qs))
print("the length of ans is ", len(ans))
print("the length of tag is ", len(tag))
print("the length of context is ", len(cntxt))

print("the first part of qs is ", qs[0])

some_df = pd.DataFrame(
    {'tag': tag, 'patterns': qs, 'responses': ans, 'context': cntxt})

print("some dataframe is ", some_df.head())

some_df.to_csv("mo_data.csv")

nextfile = 'qanda.csv'

qanda_df = pd.read_csv(nextfile, header='infer')

print(qanda_df.head())


# Function to convert a CSV to JSON
# Takes the file paths as arguments
def make_json(csvFilePath, jsonFilePath):

    # create a dictionary
    data = {}

    # Open a csv reader called DictReader
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)

        # Convert each row into a dictionary
        # and add it to data
        for rows in csvReader:

            # Assuming a column named 'No' to
            # be the primary key
            key = rows['tag']
            data[key] = rows

    # Open a json writer, and use the json.dumps()
    # function to dump data
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))

# Driver Code


# Decide the two file paths according to your
# computer system
csvFilePath = r'mo_data.csv'
jsonFilePath = r'Names.json'

make_json(csvFilePath, jsonFilePath)


def js_to_csv(in_js, out_csv):
    with open(in_js) as json_file:
        data = json.load(json_file)
    example_out = data['intents']
    csv_file = open(out_csv, 'w')
    csv_writer = csv.writer(csv_file)
    count = 0
    for tag in example_out:
        if count == 0:
            header = tag.keys()
            csv_writer.writerow(header)
            count += 1
        csv_writer.writerow(tag.values())

    csv_file.close()


in_js_lst = ['general_intents.json', 'intents_1.json', 'intents_cbot.json',
             'intents.json', 'squad_train-v2.0.json']

out_csv_lst = ['general_intents.csv', 'intents_1.csv', 'intents_cbot.csv',
               'intents.csv', 'squad_train.csv']

for j in range(len(in_js_lst)):
    print("input file is : ", in_js_lst[j])
    js_to_csv(in_js_lst[j], out_csv_lst[j])
