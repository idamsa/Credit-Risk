import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import sklearn
from uszipcode import SearchEngine


credit = pd.read_csv(r"C:\Users\Dell\Documents\Python Credit Risk\SBAnational.csv", low_memory=False)

# Inspecting dataset
print(credit.head())
print(credit.describe())
print(credit.info())
print(credit.columns)

# information  about the features in the pdf we received/ they added some other features to the data set like recession,
# percentage of loan by SBA, loans backed by real estate, new vs existent business
# also we will have to deal with all the missing values and impute them somehow or delete them
# we will have to dummy the features like y/n to 0/1 and also the target to 0/1 meaning default/paid in full
# we will have to deal with all the dates and use them to build the other column and discard them after
# also city/ state/ zip describe same variable kinda same bank / bank state
# NAICS is type of business so we might need it

credit.isnull().sum() # some have many nas we have to fix

# see the distribution of some of the features
Counter(credit['MIS_Status'])  # unbalanced + nas

Counter(credit['NewExist'])  # 136 nas and values 2=new, 1=exist, 0 we don't know what it means so we add to na
# This feature might be important so we will fix the nas and keep it
Counter(credit['State'])   # looks ok, 14 na we can fill using the ZIP
credit['Bank'].nunique()  # a lot of different values for this variable we will delete it also might not be relevant
credit['NAICS'].nunique()  # a lot of them but it says something about the data so we keep it + NO NA
credit['BankState'].nunique()  # looks ok but we will drop it because doesn't say much about the customer

# Name and id is unique and useless and we can drop entire column.
# we also delete the city (because we have state and zip), Bank Name (many values),
credit = credit.drop(columns=['LoanNr_ChkDgt', 'City', 'Name', 'Bank', 'BankState'])

# Fixing state nas

missing_rows = credit[credit['State'].isnull()].index
search = SearchEngine()

# impute State using search.by_zipcode function
for i in missing_rows:
    zipcode = search.by_zipcode(credit.iloc[i, 1])
    credit.iloc[i, 0] = zipcode.state

# Check how our NA's was imputed. We still have 2 NAs. One zipcode = 0
# and other is not in list the of search.by_zipcode function. We will remove them
credit = credit.dropna(how='any', subset=['State'])

# we have missing values in mis_status (our target variable). We will remove the rows in question because we cannot
# impute them in a way that we are 100 % that it will be representative , also they are not that many
credit = credit.dropna(how='any', subset=['MIS_Status'])

# We have to fix the other nas (esp new exist that we will use)
# We have to decide which features we keep as predictors and which we don't and if we need to build new features

