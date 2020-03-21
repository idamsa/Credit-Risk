from collections import Counter
import pandas as pd
from sklearn.utils import resample

# bring data
credit = pd.read_csv(r"C:\Users\Dell\Documents\Python Credit Risk\Data\credit_preprocessed.csv.csv", low_memory=False,
                           index_col=0)

# Histogram of the label
# Class imbalance we have many more of 1 (paid) than not paid loans.
# We will try fix this by downsampling the majority class
credit['MIS_Status'].value_counts().plot(kind='bar')

# Fixing the imbalance by downsampling the minority class

# Separate majority and minority classes
credit_majority = credit[credit.MIS_Status == 1]
credit_minority = credit[credit.MIS_Status == 0]

# Downsample majority class
credit_majority_downsampled = resample(credit_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=140758,  # to match minority class
                                       random_state=123)  # reproducible results

# Combine minority class with downsampled majority class
credit_downsampled = pd.concat([credit_majority_downsampled, credit_minority])

# Display new class counts
Counter(credit_downsampled['MIS_Status'])  # Counter({1: 140758, 0: 140758})

credit_downsampled['MIS_Status'].value_counts().plot(kind='bar')  # class imbalance fixed

# One Hot encoding categorical/ other values
state_dummy = pd.get_dummies(credit_downsampled['State'])
portion_dummy = pd.get_dummies(credit_downsampled['Portion'])
grappv_dummy = pd.get_dummies(credit_downsampled['GrAppv'])
naics_dummy = pd.get_dummies(credit_downsampled['NAICS'])

# Remove the dummyfied columns in order to append them after
credit_downsampled = credit_downsampled.drop(columns=['State', 'Portion', 'GrAppv', 'NAICS'])

# create the dummified data
credit_dummy = pd.concat([credit_downsampled, naics_dummy], axis=1)
credit_dummy = pd.concat([credit_dummy, grappv_dummy], axis=1)
credit_dummy = pd.concat([credit_dummy, portion_dummy], axis=1)
credit_dummy = pd.concat([credit_dummy, state_dummy], axis=1)

# save data
credit_dummy.to_csv('credit_downsampled_dummy.csv')