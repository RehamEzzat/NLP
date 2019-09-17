import pandas as pd

counter = pd.read_csv("counter.csv")

columns_names = counter.columns.values

rows = counter.iloc

data = pd.DataFrame(0.0, index=counter["السابق"], columns=columns_names)

for i in range(len(counter["السابق"])):
    temp = rows[i][1:]
    sum = temp.sum()
    if sum != 0:
        for c in columns_names[1:]:
            prob = (float)(rows[i][c]) / (float)(sum)
            data[c][i] = prob

data.to_csv("probabilities.csv")

