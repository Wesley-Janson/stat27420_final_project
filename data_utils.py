import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from parameters import cts_vars, categorical_vars, other_vars, confounder_vars

categorical_vars = list(categorical_vars.keys())


def read_data(data_path):
    data = pd.read_csv(data_path,dtype=object)
    # now convert types:
    data[cts_vars] = data[cts_vars].astype(float)
    data[categorical_vars] = data[categorical_vars].astype(str).replace('nan', np.nan)
    data[other_vars] = data[other_vars].astype(str)
    data["date"] = pd.to_datetime(data.date, format="%Y-%m-%d")
    return data


def prep_features(
    data, regression=False, cts_vars=cts_vars, categorical_vars=categorical_vars,
    confounders=confounder_vars):

    # outcome var -> cts if regression
    if regression: 
        data['durable_purchase']=data['durable_purchase'].replace(to_replace={
            'Good':1,'Neutral':0,'Bad':-1,"Don't know":np.nan,"Refused":np.nan})
        print(f'Excluding {len(data[data.durable_purchase.isnull()])} observations that did not answer durable purchase question.')
        data = data[data.durable_purchase.notnull()]  # require outcome
        data["durable_purchase"] = np.where(data.durable_purchase==1, 2, 
            np.where(data.durable_purchase==0,1,0))
        data = data.dropna(subset=['durable_purchase'])
        categorical_vars = [var for var in categorical_vars if var != 'durable_purchase']
    
    # normalize cts variables
    data[cts_vars] = StandardScaler().fit_transform(data[cts_vars])

    # one-hot encode categorical variables, dropping first iff regression is True
    data = pd.get_dummies(data, columns=categorical_vars, drop_first=regression)

    # prepare treatment and confounder var lists with dummies
    print(f'Excluding {len(data[data.price_change_amt_next_yr.isnull()])} observations that did not answer price change amount question.')
    data = data[data.price_change_amt_next_yr.notnull()]  # require treatment
    treatment_vars = [var for var in data.columns if 'bins' in var]
    confounder_vars = []
    for var in confounders:
        for dummy in data.columns:
            if var in dummy:
                confounder_vars.append(dummy)

    return data, treatment_vars, confounder_vars