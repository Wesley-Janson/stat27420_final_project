import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
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
    data, regression=False, missing_values='retain',
    cts_vars=cts_vars, categorical_vars=categorical_vars, confounders=confounder_vars):

    # prep outcome
    data['durable_purchase']=data['durable_purchase'].replace(to_replace={
        'Good':1,'Neutral':0,'Bad':-1,"Don't know":np.nan,"Refused":np.nan})
    # note: we could theoretically predict "don't know" for ML models, but there are so few we'll just exclude
    # (could change in future)
    print(f'Excluding {len(data[data.durable_purchase.isnull()])} observations' +
        " that refused or didn't know durable purchase question.")
    data = data[data.durable_purchase.notnull()]  # require outcome
    categorical_vars = [var for var in categorical_vars if var != 'durable_purchase']
    if not regression:
        data["durable_purchase"] += 1  # code as {0,1,2} for XGBoost

    # handle missing values
    if missing_values == 'drop':  # xgboost can handle missing values, others mostly just drop them
        temp = len(data)
        data = data.dropna(subset=confounders)
        print(f'Excluding {temp-len(data)} observations that did not answer confounder questions.')
    elif missing_values == 'impute by median':  # fast but not great
        for var in cts_vars:
            if data[var].isnull().sum()>0 and var in confounders:
                print(f'Imputing {data[var].isnull().sum()} missing values for {var} with median.')
                data[var+'_imputed'] = data[var].isnull().astype(int)
                data[var] = data[var].fillna(data[var].median())
    elif missing_values == 'impute by knn':  # very slow
        print(f'Imputing missing values via KNN.')
        imputer = KNNImputer(n_neighbors=5)
        data[cts_vars] = imputer.fit_transform(data[cts_vars])
    
    # prep cts variables
    data[cts_vars] = StandardScaler().fit_transform(data[cts_vars])

    # one-hot encode categorical variables, dropping first iff regression is True
    data[categorical_vars] = data[categorical_vars].fillna('Missing')  # new class for missing
    data = pd.get_dummies(data, columns=categorical_vars, drop_first=regression)

    # prepare treatment and confounder var lists with dummies
    print(f'Excluding {len(data[data.price_change_amt_next_yr.isnull()])} observations' +
        ' that did not answer price change amount question.')
    data = data[data.price_change_amt_next_yr.notnull()]  # require treatment
    treatment_vars = [var for var in data.columns if 'bins' in var]
    confounder_vars = []
    for var in confounders:
        for dummy in data.columns:
            if var in dummy:
                confounder_vars.append(dummy)

    return data[other_vars+confounder_vars+treatment_vars+[
        "price_change_amt_next_yr","durable_purchase"]], treatment_vars, confounder_vars


def evaluate_predictions(model, X_train, X_test, y_train, y_test, regression = False):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    if regression:
        y_pred['class'] = y_pred.idxmax(axis=1)
        y_pred = y_pred['class']
        y_pred_train['class'] = y_pred_train.idxmax(axis=1)
        y_pred_train = y_pred_train['class']
    else:
        test_predictions = [round(value) for value in y_pred]
        train_predictions = [round(value) for value in y_pred_train]
    print("Baseline accuracy: %.2f%%" % (y_test.value_counts(normalize=True).max()*100))
    print("Train accuracy: %.2f%%" % (accuracy_score(y_train, train_predictions) * 100.0))
    print("Test accuracy: %.2f%%" % (accuracy_score(y_test, test_predictions) * 100.0))
    print('\nTest predictions vs actual:')
    return pd.DataFrame({'actual': y_test, 'predicted': test_predictions}).groupby(['actual','predicted']).size()