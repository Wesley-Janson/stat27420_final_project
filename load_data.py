### Data Merging and Initial Preprocessing


## Imports
import numpy as np
import pandas as pd
import mock
from openpyxl.reader import excel


## Variables
var_renames = {
    "DATE": "date",
    'FEDFUNDS': 'fed_funds_rate',
    'UNRATE': 'unemployment_rate',
    'CPIAUCSL': 'cpi',
    'CUSR0000SAD': 'cpi_durable',
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
    }

cts_vars = [
    "consumer_sentiment_index","economic_conditions_index","consumer_expectations_index",
    "income_change_amt_next_yr","price_change_amt_next_yr","price_change_amt_next_5yr",
    "age","household_size", "n_cars", "real_income_increase_chance_next_5yr",'fed_funds_rate',
    'unemployment_rate','cpi','cpi_durable','home_price_index','1yr_inflation_via_gdp',
    '1yr_inflation_via_cpi','10yr_inflation_via_cpi'
    ]

other_vars = ["date", "case_id", "interview_id", "prev_interview_id", "prev_interview_date"]
construction_vars = ["PAGOR1", "PAGOR2", "NUMKID", "NUMADT"]


## Load and merge data

# Read in "base" Michigan CFE survey data
base_data = pd.read_csv("../paper_replication_data/MichiganConsumerSurvey.csv",dtype=object)
base_data["DATE"] = pd.to_datetime(base_data.YYYYMM, format="%Y%m").dt.strftime("%Y-%m-%d")

# Read in Federal Funds Rate from FRED
fed_funds = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=FEDFUNDS&scale=le")

# Read in Unemployment Rate from BLS via FRED
unemp = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=UNRATE&scale=left")

# Read in Headline CPI from BLS via FRED
cpi = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=CPIAUCSL&scale=le")
cpi["CPIAUCSL"] = pd.Series(cpi["CPIAUCSL"]).pct_change(12)*100

# Read in CPI Durable Goods from BLS via FRED
cpi_dur = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=CUSR0000SAD&scale")
cpi_dur["CUSR0000SAD"] = pd.Series(cpi_dur["CUSR0000SAD"]).pct_change(12)*100

# Read in Case-Shiller Index from FRED
case = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=SPCS10RSA&scale=l")

# Read in SPF 1-year ahead CPI from SPF
# Excel is corrupted (invalid 'modified' property) - fix via: https://foss.heptapod.net/openpyxl/openpyxl/-/issues/1659
with mock.patch.object(excel.ExcelReader, 'read_properties', lambda self: None):
    spf = pd.read_excel(
        "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/survey-of-professional-forecasters/historical-data/inflation.xlsx?la=en&hash=F9C3E76769B4586C3E36E403DFA54BDC",
        dtype={'YEAR': str, 'QUARTER': str})
# first datetime range from Jan 1970 to Dec 2022
date_range = pd.date_range(start="1970-01-01", end="2022-12-31", freq="MS")
dates = pd.DataFrame({"YEAR": date_range.year, "MONTH": date_range.month})
# now add quarters
dates["QUARTER"] = dates.MONTH.apply(lambda x: (x - 1) // 3 + 1).astype(str)
dates["YEAR"] = dates.YEAR.astype(str)
dates["MONTH"] = dates.MONTH.astype(str)
dates["DATE"] = dates.YEAR + "-" + dates.MONTH.str.zfill(2) + "-01"
# merge to replicate quarterly reading for each month
spf = pd.merge(dates, spf, on=["YEAR", "QUARTER"], how="left").drop(columns=["YEAR","MONTH","QUARTER"])

# merge all data sources
other_data = [fed_funds, unemp, cpi, cpi_dur, case, spf]
for i in other_data:
    base_data = base_data.merge(i, how="left", on="DATE")
base_data["DATE"] = pd.to_datetime(base_data.DATE, format="%Y-%m-%d")

# initial preprocessing

# first convert all to string, strip all, recode '' to np.nan
base_data[list(var_renames.keys())+construction_vars] = base_data[
    list(var_renames.keys())+construction_vars
    ].astype(str).apply(lambda x: x.str.strip()
    ).replace('', np.nan).replace('nan', np.nan)
base_data.rename(columns=var_renames,inplace=True)

# now recode categorical variables
base_data["first_interview"] = np.where(base_data.prev_interview_id.isna(), 1, 0)
base_data["price_related_yr_ago"] = np.where(
    (base_data.PAGOR1.astype(str).isin(["14","54"]))|(base_data.PAGOR2.astype(str).isin(["14","54"])), 1, 0)
for var,codes in categorical_vars.items():
    base_data[var] = base_data[var].astype(float).replace(categorical_vars[var])

# convert cts vars to numeric
base_data["household_size"] = base_data.NUMKID.astype(float) + base_data.NUMADT.astype(float)
base_data[cts_vars] = base_data[cts_vars].astype(float)
base_data["price_change_amt_next_5yr"] = base_data.PX5Q2.replace([98,99], np.nan)

base_data = base_data.drop(columns=construction_vars)