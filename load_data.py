### Data Merging and Initial Preprocessing


## Imports
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import mock
from openpyxl.reader import excel
from parameters import var_renames, categorical_vars, cts_vars, other_vars, construction_vars


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
cpi["CPIAUCSL_1mo_lag"] = cpi["CPIAUCSL"].shift(1)

# Read in CPI Durable Goods from BLS via FRED
cpi_dur = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=CUSR0000SAD&scale")
cpi_dur["CUSR0000SAD"] = pd.Series(cpi_dur["CUSR0000SAD"]).pct_change(12)*100
cpi_dur["CUSR0000SAD_1mo_lag"] = cpi_dur["CUSR0000SAD"].shift(1)

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


## Initial preprocessing

# drop columns we don't need
data = base_data[list(var_renames.keys())+construction_vars]

# first convert all to string, strip all, recode '' to np.nan
data = data.astype(str).apply(lambda x: x.str.strip()
    ).replace('', np.nan).replace('nan', np.nan)
data.rename(columns=var_renames,inplace=True)

# convert cts vars to numeric
data["household_size"] = data.NUMKID.astype(float) + data.NUMADT.astype(float)
data[cts_vars] = data[cts_vars].astype(float)
print(f'Excluding {len(data[(data.price_change_next_yr!="8")&(data.price_change_next_yr!="9")])} observations that did not answer 1 year price change question.')
data = data[(data.price_change_next_yr!="8")&(data.price_change_next_yr!="9")]
data[["price_change_amt_next_yr","price_change_amt_next_5yr"]] = data[[
    "price_change_amt_next_yr","price_change_amt_next_5yr"]].fillna(0)
data["price_change_amt_next_yr"] = data.price_change_amt_next_yr.replace(
    to_replace={98:np.nan,99:0})  # 98 = don't know, 99 = NA
data["price_change_amt_next_5yr"] = data.price_change_amt_next_5yr.replace(
    to_replace={98:np.nan,99:0})  # 98 = don't know, 99 = NA)

# create two different treatment variables
data['pctiles'] = np.nan
for i in data.date.unique():
    data['pctiles'] = np.where(data['date']==i, data.price_change_amt_next_yr.rank(pct = True), data['pctiles']) 
data["treatment_pctile"] = pd.cut(data['pctiles'],
                      bins=[0.0, 0.2, 0.4, 0.6, 0.8, float('Inf')],
                      labels=[1, 2, 3, 4, 5])
data.drop(columns=['pctiles'])
data["treatment_bins"] = pd.cut(data['price_change_amt_next_yr'],
                      bins=[-0.000001, 5, 10, 15, 20, float('Inf')],
                      labels=[1, 2, 3, 4, 5])

# Create ZLB Variable
data["zlb"] = np.where(data.FEDFUNDS < 0.25, 1, 0)

# now recode categorical variables
data["first_interview"] = np.where(data.prev_interview_id.isna(), 1, 0)
data["price_related_yr_ago"] = np.where(
    (data.PAGOR1.astype(str).isin(["14","54"]))|(data.PAGOR2.astype(str).isin(["14","54"])), 1, 0)
for var,codes in categorical_vars.items():
    data[var] = data[var].astype(float).replace(categorical_vars[var])

data = data.drop(columns=construction_vars)
categorical_vars = list(categorical_vars.keys())