### Data Merging

import numpy as np
import pandas as pd
import mock
from openpyxl.reader import excel

# Read in "base" Michigan CFE survey data
base_data = pd.read_csv("../paper_replication_data/MichiganConsumerSurvey.csv")
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

##### Merge All Data Together
other_data = [fed_funds, unemp, cpi, cpi_dur, case, spf]
for i in other_data:
    base_data = base_data.merge(i, how="left", on="DATE")