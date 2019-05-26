import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('dataset/Market_Basket_Optimisation.csv', header = None)

from apyori import apriori

transactions = []

for i in range(0, 7501):
    transactions.append(list(dataset.iloc[i, :]))


results = list(apriori(test))

#del(cleanedList)
test = []

for i in range(7501):
    cleanedList = [x for x in transactions[i] if str(x) != 'nan']
    test.append(cleanedList)

