import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from parameters import cts_vars, categorical_vars, other_vars
categorical_vars = list(categorical_vars.keys())


def prep_features(data, regression=False, cts_vars=cts_vars, categorical_vars=categorical_vars):
    
    # normalize cts variables
    data[cts_vars] = StandardScaler().fit_transform(data[cts_vars])

    # one-hot encode categorical variables, dropping first iff regression is True
    data = pd.get_dummies(data, columns=categorical_vars, drop_first=regression)

    return data