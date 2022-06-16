# Visualizing the CO2 emissions training datase
# 
# Learner: Nguyen Truong Thinh
# Contact me: nguyentruongthinhvn2020@gmail.com || +84393280504
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
# Loading the data from a CSV file.
co2 = pd.read_csv("CO2_subset.csv")
co2 = pd.get_dummies(co2, columns=['Ft'], prefix='', prefix_sep='')

# Extract a training and a testing dataset from the whole data.
train_dataset = co2.sample(frac=0.8, random_state=0)
test_dataset = co2.drop(train_dataset.index)
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('Enedc (g/km)')
test_labels = test_features.pop('Enedc (g/km)')
test_results = {}

# %%
# Visualising the training data set.
sns.pairplot(train_dataset[['m (kg)', 'Enedc (g/km)', 'ec (cm3)']], diag_kind='kde')
plt.show()