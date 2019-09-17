import re
import pandas as pd
from collections import OrderedDict

input_string = input()
input_list = re.split(" ", input_string)


probability = pd.read_csv("probabilities.csv")
length = len(probability["السابق"])

first_place_indices = []
second_place_indices = []


for i in range(length):

    if eval(probability["السابق"][i])[0] == input_list[-1]:
        first_place_indices.append(i)
    elif eval(probability["السابق"][i])[1] == input_list[-1]:
        second_place_indices.append(i)

possibilities = []
dictionary = {}
columns_names = probability.columns.values
for column_name in columns_names:
    if column_name == "السابق":
        continue
    for index_1 in first_place_indices:
        if probability[column_name][index_1] > 0:
            temp_str = eval(probability["السابق"][index_1])[1] + column_name
            dictionary[temp_str] = probability[column_name][index_1]
    for index_2 in second_place_indices:
        if probability[column_name][index_2] > 0:
            dictionary[column_name] = probability[column_name][index_2]

print(dictionary)
# ascending
a = OrderedDict(sorted(dictionary.items(), key=lambda kv: kv[1]))

# descending
d = OrderedDict(sorted(dictionary.items(), key=lambda kv: kv[1], reverse= True))

print(d)
l = list(d.items())

print("do you mean: ", input_list[-1], l[0][0])
