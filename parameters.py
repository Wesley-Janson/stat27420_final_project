import numpy as np


## Data variables

var_renames = {
    "DATE": "date",
    'FEDFUNDS': 'fed_funds_rate',
    'UNRATE': 'unemployment_rate',
    'CPIAUCSL': 'cpi',
    "CPIAUCSL_1mo_lag": "cpi_1mo_lag",
    'CUSR0000SAD': 'cpi_durable',
    'CUSR0000SAD_1mo_lag': 'cpi_durable_1mo_lag',
    'SPCS10RSA': 'home_price_index',
    'INFPGDP1YR': '1yr_inflation_via_gdp',
    'INFCPI1YR': '1yr_inflation_via_cpi',
    'INFCPI10YR': '10yr_inflation_via_cpi',
    "CASEID": "case_id",
    "ID": "interview_id",
    "IDPREV": "prev_interview_id",
    "DATEPR": "prev_interview_date",
    "ICS":"consumer_sentiment_index",
    "ICC":"economic_conditions_index",
    "ICE":"consumer_expectations_index",
    "PAGO":"personal_finances_yr_ago",
    "PAGO5":"personal_finances_5yr_ago",
    "PEXP":"personal_finances_next_yr",
    "PEXP5":"personal_finances_next_5yr",
    "INEXQ1":"income_change_next_yr",
    "INEXQ2":"income_change_amt_next_yr",
    "RINC":"real_income_expectations",
    "BAGO":"conditions_yr_ago",
    "BEXP":"conditions_next_yr",
    "UNEMP":"unemployment_next_yr",
    "GOVT":"govt_policy_efficacy",
    "RATEX":"interest_rates_next_yr",
    "PX1Q1":"price_change_next_yr",
    "PX1Q2":"price_change_amt_next_yr",
    "PX5Q1":"price_change_next_5yr",
    "PX5Q2":"price_change_amt_next_5yr",
    "DUR":"durable_purchase",
    "CAR":"car_purchase",
    "YTL5":"income_quintile",
    "AGE":"age",
    "REGION":"region",
    "SEX":"sex",
    "EDUC":"education",
    "VEHOWN":"vehicle_ownership",
    "VEHNUM":"n_cars",
    "PINC":"real_income_increase_chance_next_5yr",
    }

categorical_vars = {
    "first_interview": {},
    "personal_finances_yr_ago": {1: "Better", 3: "Same", 5: "Worse", 8: "Don't know", 9: "Refused"},
    "price_related_yr_ago": {},
    "personal_finances_5yr_ago": {1: "Better", 3: "Same", 5: "Worse", 8: "Don't know", 9: "Refused"},
    "personal_finances_next_yr": {1: "Better", 3: "Same", 5: "Worse", 8: "Don't know", 9: "Refused"},
    "personal_finances_next_5yr": {1: "Better", 3: "Same", 5: "Worse", 8: "Don't know", 9: "Refused"},
    "income_change_next_yr": {1: "Higher", 3: "Same", 5: "Lower", 8: "Don't know", 9: "Refused"},
    "real_income_expectations": {1: "Higher", 3: "Same", 5: "Lower", 8: "Don't know", 9: "Refused"},
    "conditions_yr_ago": {1: "Better", 3: "Same", 5: "Worse", 8: "Don't know", 9: "Refused"},
    "conditions_next_yr": {1: "Better", 3: "Same", 5: "Worse", 8: "Don't know", 9: "Refused"},
    "unemployment_next_yr": {1: "Higher", 3: "Same", 5: "Lower", 8: "Don't know", 9: "Refused"},
    "govt_policy_efficacy": {1: "Good", 3: "Fair", 5: "Poor", 8: "Don't know", 9: "Refused"},
    "interest_rates_next_yr": {1: "Higher", 3: "Same", 5: "Lower", 8: "Don't know", 9: "Refused"},
    "price_change_next_yr": {1: "Higher", 2:"Higher same rate", 3: "Same", 5: "Lower", 8: "Don't know", 9: "Refused"},
    "price_change_next_5yr": {1: "Higher", 2:"Higher same rate", 3: "Same", 5: "Lower", 8: "Don't know", 9: "Refused"},
    "durable_purchase": {1: "Good", 3: "Neutral", 5:"Bad", 8: "Don't know", 9: "Refused"},
    "car_purchase": {1: "Good", 3: "Neutral", 5:"Bad", 8: "Don't know", 9: "Refused"},
    "income_quintile": {1: "Lowest", 2: "Lower middle", 3: "Middle", 4: "Upper middle", 5: "Highest"},
    "region": {1: "West", 2: "North Central", 3: "Northeast", 4: "South",6: np.nan},
    "sex":{1: "Male", 2: "Female"},
    "education": {1: "No high school", 2: "Partial high school", 3: "High school", 4: "Some college", 5: "College", 6: "Graduate school"},
    "vehicle_ownership": {1: "Yes", 5: "No", 8: "Don't know", 9: "Refused"},
    "treatment_pctile": {1: "Lowest Group", 2: "Low Group", 3: "On Target", 4: "High Group", 5: "Highest Group"},
    "treatment_bins": {1: "0-5", 2: "5-10", 3: "10-15", 4: "15-20", 5: "20+"}
    }

cts_vars = [
    "consumer_sentiment_index","economic_conditions_index","consumer_expectations_index",
    "income_change_amt_next_yr","price_change_amt_next_yr","price_change_amt_next_5yr",
    "age","household_size", "n_cars", "real_income_increase_chance_next_5yr",'fed_funds_rate',
    'unemployment_rate','cpi','cpi_1mo_lag','cpi_durable','cpi_durable_1mo_lag','home_price_index',
    '1yr_inflation_via_gdp','1yr_inflation_via_cpi','10yr_inflation_via_cpi'
    ]

other_vars = ["date", "case_id", "interview_id", "prev_interview_id", "prev_interview_date"]
construction_vars = ["PAGOR1", "PAGOR2", "NUMKID", "NUMADT"]

confounder_vars = [
    "fed_funds_rate",
    "unemployment_rate",
    "cpi_1mo_lag",
    "cpi_durable_1mo_lag",
    "personal_finances_next_yr",
    "income_change_amt_next_yr",
    "conditions_next_yr",
    "unemployment_next_yr",
    "income_quintile",
    "age",
    "sex",
    "education",
    "household_size",
    "price_related_yr_ago"
]