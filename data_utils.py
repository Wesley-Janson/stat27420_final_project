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
    """missing_values options: 'retain all', 'retain cts', 'impute by knn', 'impute by median',
        'drop cts', 'drop all'"""

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
        data["durable_purchase"] = data["durable_purchase"].astype(int)

    # handle missing values
    # xgboost can handle missing values, others mostly just drop them
    if missing_values == 'drop cts' or missing_values == 'drop all': 
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
    if missing_values == 'drop all':
        temp = len(data)
        data[categorical_vars] = data[categorical_vars].dropna()
        print(f'Excluding {temp-len(data)} observations that did not answer confounder questions.')
    elif missing_values != 'retain all':
        data[categorical_vars] = data[categorical_vars].fillna('Missing')  # new class for missing
    
    data_treatment_bins = data[data.price_change_amt_next_yr.notnull()]
    data_treatment_bins = data_treatment_bins["treatment_bins"]
    
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

    int_data = data[other_vars+confounder_vars+treatment_vars+[
        "price_change_amt_next_yr","durable_purchase"]]
    int_data["treatment_bins"] = data_treatment_bins
    int_data["treatment_bins"] = np.where(int_data["treatment_bins"]=="0-5",0,
                                        np.where(int_data["treatment_bins"]=="5-10",1,
                                        np.where(int_data["treatment_bins"]=="10-15",2,
                                        np.where(int_data["treatment_bins"]=="15-20",3,4))))

    return int_data, treatment_vars, confounder_vars


def summarize_predictions(y_train, train_predictions, y_test, test_predictions):
    print("Baseline accuracy: %.2f%%" % (y_test.value_counts(normalize=True).max()*100))
    print("Train accuracy: %.2f%%" % (accuracy_score(y_train, train_predictions) * 100.0))
    print("Test accuracy: %.2f%%" % (accuracy_score(y_test, test_predictions) * 100.0))
    print('\nTest predictions vs actual:')
    if type(y_test) == list:
        y_test = pd.Series(y_test)
        test_predictions = pd.Series(test_predictions)
    y_test = y_test.rename('actual').reset_index()
    test_predictions = test_predictions.rename('predicted').reset_index()
    return pd.concat([y_test, test_predictions],axis=1
    ).groupby(['actual','predicted']).size()


def evaluate_predictions(model, X_train, X_test, y_train, y_test, regression = False):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    if regression:
        y_pred['class'] = y_pred.idxmax(axis=1)-1
        test_predictions = y_pred['class']
        y_pred_train['class'] = y_pred_train.idxmax(axis=1)-1
        train_predictions = y_pred_train['class']
    else:
        test_predictions = [round(value) for value in y_pred]
        train_predictions = [round(value) for value in y_pred_train]
    return summarize_predictions(y_train, train_predictions, y_test, test_predictions)


def rebin_outcome(data):
    if data['durable_purchase'].min() == -1:
        regression = True
        data['durable_purchase'] += 1
    else:
        regression = False
    data['durable_good'] = np.where(data['durable_purchase']==2,1,0)
    data['durable_bad'] = np.where(data['durable_purchase']==0,1,0)
    if regression:
        data['durable_purchase'] -= 1
    return data


def unrebin_outcome(ya_pred,yb_pred):
    predictions = pd.DataFrame(data={'isgood':ya_pred,'isbad':yb_pred})
    predictions["durable_purchase"] = np.where(
        (predictions["isgood"] == 0) & (predictions["isbad"] == 1), 0, -1)
    predictions["durable_purchase"] = np.where(
        (predictions["isgood"] == 0) & (predictions["isbad"] == 0), 1, predictions["durable_purchase"])
    predictions["durable_purchase"] = np.where(
        (predictions["isgood"] == 1) & (predictions["isbad"] == 0), 2, predictions["durable_purchase"])
    return predictions.durable_purchase


def evaluate_multilevel_predictions(modela, modelb, X_train, X_test, y_train, y_test):
    ya_pred = modela.predict(X_test)
    yb_pred = modelb.predict(X_test)
    ya_pred_train = modela.predict(X_train)
    yb_pred_train = modelb.predict(X_train)
    test_predictions = unrebin_outcome(ya_pred,yb_pred)
    train_predictions = unrebin_outcome(ya_pred_train,yb_pred_train)
    return(summarize_predictions(
        y_train.durable_purchase, train_predictions, 
        y_test.durable_purchase, test_predictions))