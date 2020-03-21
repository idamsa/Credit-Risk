import math
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from uszipcode import SearchEngine

# bring data
credit = pd.read_csv(r"C:\Users\Dell\Documents\Python Credit Risk\Data\SBAnational.csv", low_memory=False)

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

credit.isnull().sum()  # some have many nas we have to fix

# see the distribution of some of the features
Counter(credit['MIS_Status'])  # unbalanced + nas

Counter(credit['NewExist'])  # 136 nas and values 2=new, 1=exist, 0 we don't know what it means so we add to na
# This feature might be important so we will fix the nas and keep it
Counter(credit['State'])  # looks ok, 14 na we can fill using the ZIP
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

# see the distribution of target feature
Counter(credit['MIS_Status'])  # unbalanced + nas Counter({'P I F': 739609, 'CHGOFF': 157558, nan: 1997})

# Fixing state nas
missing_rows = credit[credit['State'].isnull()].index
search = SearchEngine()

# impute State using zearch.by_zipcode function
for i in missing_rows:
    zipcode = search.by_zipcode(credit.iloc[i, 1])
    credit.iloc[i, 0] = zipcode.state

# Check how our NA's was imputed. We still have 2 NAs. One zipcode = 0
# and other is not in list the of search.by_zipcode function. We will remove them
credit = credit.dropna(how='any', subset=['State'])

# we have missing values in mis_status (our target variable). We will remove the rows in question because we cannot
# impute them in a way that we are 100 % that it will be prepresentative , also they are nit that many
credit = credit.dropna(how='any', subset=['MIS_Status'])
# Change labels target feature
credit.loc[credit['MIS_Status'] == "P I F", 'MIS_Status'] = 1  # Paid in full = 1
credit.loc[credit['MIS_Status'] == "CHGOFF", 'MIS_Status'] = 0  # Charged off = 0
credit["MIS_Status"] = credit.MIS_Status.astype(object)

# Nas is new exist and change to 0/1 label
# after change type to object we will impute nas to most frequent value
credit["NewExist"] = credit.NewExist.astype(object)
credit['NewExist'].fillna(1, inplace=True)  # fill nas with 1 (most frequent)
credit.loc[credit['NewExist'] == 0, 'NewExist'] = 1  # change the 0 to 1 (most frequent)
credit.loc[credit['NewExist'] == 2, 'NewExist'] = 0  # change 2 to 0 so 0 = new, 1 = established
credit["NewExist"] = credit.NewExist.astype(object)

# add column loans backed by “RealEstate,” where  “RealEstate” = 1 if “Term” > 240 months
# and “RealEstate” = 0 if “Term” <240 months/ Counter({0: 831027, 1: 66138})
credit['RealEstate'] = np.where(credit['Term'] > 240, 1, 0)

# fix na and relabel LowDoc Loan Program: Y = Yes, N = No to 0= no and 1 = yes # we have also other values (S,A,0,R,C,1) that
# we will also change to 1 or 0
# We will fill nas with most frequent aka N (no)
credit['LowDoc'].fillna("N", inplace=True)
credit['LowDoc'] = credit.LowDoc.replace(dict.fromkeys(['C', '1', 'S', 'A', 'R', '0'], 'N'))

# Change the label to low doc from N and Y to 0 for no and 1 for yes
credit.loc[credit['LowDoc'] == "N", 'LowDoc'] = 0
credit.loc[credit['LowDoc'] == "Y", 'LowDoc'] = 1
credit["LowDoc"] = credit.LowDoc.astype(object)

# build a column with the pecentage of sba covered of the loan(ratio of the
# amount of the loan SBA guarantees and the gross amount approved by the bank (SBA_Appv/GrAppv).)
cols_to_change = ['SBA_Appv', 'GrAppv']

for col in cols_to_change:
    credit[col] = credit[col].str[1:]
    credit[col] = credit[col].str.slice(0, -2)
    credit[col] = credit[col].replace(',', '', regex=True)
    credit[col] = credit[col].astype(float)

credit['Portion'] = round((credit['SBA_Appv'] / credit['GrAppv']) * 100)  # Build the new column
credit["Portion"] = credit.Portion.astype(int)

# NAICS is the code for the industry
# We can see we have the value 0 for 201666 of the rows

# Let's check the mis status for those rows. If they are not many party of the minority class we could delete them
NAICS_0 = credit[credit['NAICS'] == 0]
Counter(NAICS_0.MIS_Status).values()  # 184868 - 0, 16798 - 1, we can delete them
credit = credit[credit.NAICS != 0]


# We will leave just the two first numbers of the code that tells the industry(general)
# instead of the 5 numbers that are more in detail
def first_two(d):
    return (d // 10 ** (int(math.log(d, 10)) - 1))


credit["NAICS"] = credit.NAICS.astype(int)
credit['NAICS'] = credit.NAICS.apply(first_two)
credit["NAICS"] = credit.NAICS.astype(object)

sns.pairplot(credit, kind="scatter", hue="MIS_Status")
plt.show()
# After analyzing the plot above we will delete the NoEmp, Createjob, 'RetainedJob','FranchiseCode', 'UrbanRural',
# 'RevLineCr' bacause they say nothing about the label
# The 'DisbursementGross', 'BalanceGross', 'ChgOffPrinGr' they are all 3 amounts that we can get only after the end of
# the credit period so we will not use it for the prediction
# so we will delete them as they have no use in our predictions
# We can delete the zip because we already used it to fill the state
# we can delete the term because we used it for bulding the real estate column
# we can delete all the dates because they have a lot of na and we will not use them in the prediction ( 'ChgOffDate',
# 'DisbursementDate')
# We also delete sba app value because we used it to build the portion %
# Name and id are unique we drop entire column.
# we also delete the city (because we have state and zip and we dont want duplicated predictors),
# Bank Name (many values and not related to customer)

credit = credit.drop(columns=['LoanNr_ChkDgt', 'City', 'Name', 'Bank', 'BankState'])
credit = credit.drop(
    columns=['Zip', 'ChgOffDate', 'DisbursementDate', 'SBA_Appv', 'ApprovalDate', 'ApprovalFY', 'Term'])
credit = credit.drop(
    columns=['NoEmp', 'CreateJob', 'RetainedJob', 'FranchiseCode', 'UrbanRural', 'RevLineCr', 'DisbursementGross',
             'BalanceGross', 'ChgOffPrinGr'])

# check GrAppv(maybe we bin it) highly skewed to the left
plt.figure(figsize=(15, 8))
sns.distplot(credit.GrAppv, color="g", kde=False)
plt.ylabel('Density')
plt.title('Distribution of Approved ammount')
plt.show()

# binning the gr app
credit['GrAppv'] = pd.cut(credit['GrAppv'], 4)

# we will also bin the portion column
credit['Portion'] = pd.cut(credit['Portion'], 5)

# save preprocessed dataframe
credit.to_csv('credit_preprocessed.csv')
