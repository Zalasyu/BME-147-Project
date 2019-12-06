import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set()
from scipy.stats import chi2_contingency
from scipy.stats import chi2

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

desired_width = 320
pd.set_option('display.width', desired_width)

pd.set_option('display.max_columns', 10)
# ______________________________________________________________________________________________________________________
# Pulling Data

data = pd.read_csv('Turing-ish Test_December 6, 2019_12.14.csv')
# ______________________________________________________________________________________________________________________
# Data Wrangling

# Columns of interest selected.
data = data[['Q2', 'Q1']]

data.drop([0, 1], inplace=True)
data.reset_index(inplace=True)
###################
data = data[['Q2', 'Q1']]
data.columns = ['Gender', 'Pass/Fail']
###################
# Convert answers to 0 or 1
data.replace(to_replace='B', value=0, inplace=True) # All who chose B is converted to a numerical data type of 0
data.replace(to_replace='A', value=1, inplace=True) # All who chose A is converted to a numerical data type of 1
###################
# Is a Female? True = 1 False = 0
data.replace(to_replace='Male', value=0, inplace=True) # If not a female then by boolean logic it is False = 0
data.replace(to_replace='Female', value=1, inplace=True) # If a female then by boolean logic it is True = 1
# ______________________________________________________________________________________________________________________

"""Chi-Squared"""
# For testing if there is any statistical significance.
alpha = 0.05
matrix_data = [[12,16],
               [11,15]]
"""
Observed Data
________________________________________
         |   Pass  |  Fail  |   Total
Female   |    12   |   16   |    28
Male     |    11   |   15   |    26
Total    |    23   |   31   |    54 
_______________________________________
Note: A 2X2 matrix, therefore we need to do Yate's Continuity Correction
"""
# We want to get the expected values that ASSUME there is no relationship between the being FEMALE and PASSING.
pass_odds_ratio = 23/54  # This is the probability of passing
fail_odds_ratio = 31/54  # This is the probability of failing

tot_fem_pop = 28
tot_male_pop = 26

# Getting expected values, if we assume there is no relationship being female and passing the test.
expected_pass_fem = tot_fem_pop * pass_odds_ratio
print(f'The number of females passing: {expected_pass_fem}')
print(f'The number of females not passing: {28-expected_pass_fem} ')

expected_pass_male = tot_male_pop *pass_odds_ratio
print(f'The number of males passing: {expected_pass_male}')
print(f'The number of males not passing: {26-expected_pass_male} ')


"""
Expected Data
________________________________________
         |     Pass     |     Fail    |   Total
Female   |    11.926*   |   16.074*   |    28
Male     |    11.074*   |   14.926*   |    26
Total    |      23      |     31      |    54 
_______________________________________
Note: A 2X2 matrix, therefore we need to do Yate's Continuity Correction
"""
print(f'\nThis is the observed data below:\n\n{matrix_data}\n')
stat, p, dof, expected = chi2_contingency(matrix_data, correction=True)
print(f'Using scipy chi2 function below is the expected table:\n\n{expected}\n')
"""
Chi-Squared Statistic
The test statistic should be designed to describe, with a single
number, how much the “observed” frequencies differ from the
“expect” frequencies (i.e, the frequencies we would expect if the null hypothesis is true)

χ2 = ∑((O − E)^2)/E

If Statistic >= Critical Value: significant result, reject null hypothesis (H0), dependent.
If Statistic < Critical Value: not significant result, fail to reject null hypothesis (H0), independent.

degrees of freedom: (rows - 1) * (cols - 1)
"""
rows = 2
columns = 2
DOF = (rows - 1) * (columns - 1)
print(f'The Degrees of Freedom for the chi-squared distribution is: {DOF}\n')

# χ2 = ∑((O − E)^2)/E
print(f'The chi-statistic is: {stat}')  # Chi-statistic
prob = 1 - alpha
critical = chi2.ppf(prob, DOF)
print(f'The critical value is: {critical}\n')

if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)\nThere is no relationship of being female and being able to detect AI '
          'better than males.')

#
"""
Calculating expected values if there is no relationship of being female and being able to detect AI better.

The number of females passing: 11.925925925925926
The number of females not passing: 16.074074074074076 
The number of males passing: 11.074074074074074
The number of males not passing: 14.925925925925926 

################################################################

This is the observed data below:

Observed Data

[[12, 16], [11, 15]]
________________________________________
         |   Pass  |  Fail  |   Total
Female   |    12   |   16   |    28
Male     |    11   |   15   |    26
Total    |    23   |   31   |    54 
_______________________________________

################################################################

Using scipy chi2 function below is the expected table:

Expected Data

[[11.92592593 16.07407407]
 [11.07407407 14.92592593]]
________________________________________
         |     Pass     |     Fail    |   Total
Female   |    11.926*   |   16.074*   |    28
Male     |    11.074*   |   14.926*   |    26
Total    |      23      |     31      |    54 
_______________________________________

The Degrees of Freedom for the chi-squared distribution is: 1

The chi-statistic is: 0.05503367600141795
The critical value is: 3.841458820694124

Independent (fail to reject H0)
There is no relationship of being female and being able to detect AI better than males.
"""