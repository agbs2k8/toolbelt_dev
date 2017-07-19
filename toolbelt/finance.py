# -*- coding: utf-8 -*-'
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
from .utils import validate_str


def import_reference_file():
    stock_reference = pd.read_csv('C:\\Users\\ajwilson\\desktop\\Finc_Mode\\stock_reference.txt', sep = '\t', encoding = 'latin-1')
    stock_reference['exchange'] = stock_reference['Exchange:Ticker'].str.split(":").str[0]
    stock_reference['ticker'] = stock_reference['Exchange:Ticker'].str.split(":").str[1]
    stock_reference['ticker'] = stock_reference['ticker'].str.upper()
    stock_reference = stock_reference.drop(labels = ['Exchange:Ticker'], axis = 1)
    stock_reference = stock_reference.set_index('ticker')
    for col in stock_reference.columns.values:
        stock_reference[col] = pd.to_numeric(stock_reference[col], errors = 'coerce')
    return stock_reference


@validate_str
def get_key_ratios(ticker):
    key_ratios = {}
    # ensure the ticker is uppercase
    ticker = ticker.upper()

    # load expanded financial data from reference file
    try:
        stock_reference = import_reference_file().loc[ticker]
    except:
        print("Ticker Error: Ticker not in reference file.")
        return None

    # retrieve current price of the stock by ticker
    current_prices = pd.DataFrame(get_current(ticker)['Time Series (1min)']).transpose().reset_index()
    key_ratios['current_price'] = float(
        current_prices[current_prices['index'] == current_prices['index'].max()]["4. close"].iloc[0])

    # ratios
    key_ratios['pe_ratio'] = stock_reference['P_LTM Diluted EPS Before Extra']
    key_ratios['ev_ebitda'] = stock_reference['Total Enterprise Value'] / stock_reference['EBITDA']
    key_ratios['ev_bv'] = stock_reference['Total Enterprise Value'] / (stock_reference['Total Assets'] - stock_reference['Total Liabilities'])
    key_ratios['bv_share'] = (stock_reference['Total Assets'] - stock_reference['Total Liabilities']) / stock_reference['Shares Outstanding']
    key_ratios['profit_margin'] = stock_reference['Gross Margin'] / 100
    key_ratios['operating_margin'] = stock_reference['Earnings from Cont Ops Margin'] / 100
    key_ratios['roa'] = stock_reference['Return on Assets'] / 100
    key_ratios['roe'] = stock_reference['Return on Equity'] / 100
    key_ratios['debt_equity'] = stock_reference['Total Debt_Equity'] / 100

    # graham
    key_ratios['graham_value'] = (stock_reference['Diluted EPS Incl Extra Items'] * (8.5 + (2 * (stock_reference['LT EPS Growth Rate']))) * 4.4) / (
                                 (float(get_corporate_rates()['Yield'].iloc[0].split('%')[0]) / 100) * 100)
    key_ratios['graham_safety'] = 1 - (key_ratios['current_price'] / key_ratios['graham_value'])

    return key_ratios


def get_corporate_rates():
    url = "http://markets.on.nytimes.com/research/markets/bonds/bonds.asp"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    table = soup.find(class_="finra")
    subtable = table.find_all('tbody')[0]
    df_columns = ['Issuer','Cupon','Maturity','Moody','S&P','Fitch','Last','Change','Yield']
    df_lists = []
    for row in subtable.find_all('tr'):
        row_list = []
        for col in row.find_all('td'):
            row_list.append(col.text)
        df_lists.append(row_list)
    df = pd.DataFrame(df_lists, columns = df_columns)
    return df


def get_fed_rates():
    url = "https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yield"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    table = soup.find(class_='t-chart')
    df_columns = ['Date','1mo','3mo','6mo','1yr','2yr','3yr','5yr','7yr','10yr','20yr','30yr']
    df_lists = []
    # Find all the <tr> tag pairs, skip the first one, then for each.
    for row in table.find_all('tr')[1:]:
        row_list = []
        for col in row.find_all('td'):
            row_list.append(col.string.strip())
        df_lists.append(row_list)
    df=pd.DataFrame(df_lists, columns = df_columns)
    df.Date = pd.to_datetime(df.Date)
    for col in df_columns[1:]:
        df[col] = pd.to_numeric(df[col], errors = 'coerce')
    return df
# df[df.Date == df.Date.max()]['30yr'].iloc[0]


def get_inflation():
    url = "https://www.statbureau.org/en/united-states/inflation"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    table = soup.find(class_="currnet-inflation-table")
    rate_str = table.find_all('tr')[0].find_all('td')[1].string.strip()
    return float(rate_str.split('%')[0])/100


@validate_str
def get_current(ticker):
    ticker = ticker.upper()
    with open('C:\\Users\\ajwilson\\Desktop\\Finc_Mode\\alphavantage_api.json','r') as j:
        credentials = json.load(j)
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol="+ticker+"&interval=1min&apikey="+credentials['api_key']
    r = requests.get(url=url)
    return r.json()
# pd.DataFrame(get_current('MSFT')['Time Series (1min)']).transpose()
