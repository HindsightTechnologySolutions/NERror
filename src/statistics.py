####################################
### Neccessary Import Statements ###
####################################
import pandas as pd
import numpy as np

#########################################################
### Write the Class That Calculates Necessary Stats. ###
#########################################################

"""
Parameters: 
df = Pandas dataframe
Return: 
Recall score
true positive / (true positive + false negative)
"""


def recall(df):
    true_positive = df["tp"].sum()  # true positive score from dataframe
    false_negative = df["fn"].sum()  # false negative score from dataframe
    return df["tp"] / (df["tp"] + df["fn"])
