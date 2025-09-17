import pandas as pd 

data = pd.read_csv("housing.csv")


#some quick look at the data
print(data.head())
print(data.columns)
print(data['ocean_proximity'])

data['ocean_proximity'].hist()
#data.hist()


#one-hot encoding step for categorical data
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown= 'ignore')

enc = OneHotEncoder(sparse_output=False)
enc.fit(data[['ocean_proximity']])
encoded_data = enc.transform(data[['ocean_proximity']])

category_names = enc.get_feature_names_out()

encoded_data_df = pd.DataFrame(encoded_data, columns=category_names)
data = pd.concat([data, encoded_data_df], axis = 1)

data = data.drop(columns = 'ocean_proximity')

#data.to_csv("revised_data.csv")


import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

#Data Splitting:

data["income_categories"] = pd.cut(data["median_income"],
                               bins=[0,2,4,6, np.inf],
                               labels=[1,2,3,4])

my_splitter = StratifiedShuffleSplit(n_splits = 1,
                                    test_size = 0.2,
                                    random_state = 42)

for train_index, test_index in my_splitter.split(data, data["income_categories"]):
    strat_data_train = data.loc[train_index].reset_index(drop=True)
    strat_data_test = data.loc[test_index].reset_index(drop=True)
strat_data_train = strat_data_train.drop(columns=["income_categories"], axis=1)
strat_data_test = strat_data_test.drop(columns=["income_categories"], axis=1)

print(data.shape)
print(strat_data_train.shape)
print(strat_data_test.shape)


from sklearn.preprocessing import StandardScaler
my_scaler = StandardScaler()
my_scaler.fit(strat_data_train.iloc[:,0:-5])
scaled_data_train = my_scaler.transform(X_train.iloc[:,0:-5])
scaled_data_train_df = pd.DataFrame(scaled_data_train, columns=strat_data)
X_train = scaled_data_train_df.join(strat_data_train.iloc[:,-5:])



corr_matrix = data.corr()
corr_matrix.to_csv('corr.csv')

print("correlation matrix")

print(corr_matrix)

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(np.abs(corr_matrix))



