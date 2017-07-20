# -*- coding: utf-8 -*-'
import os
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
from .utils import validate_str

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def import_reference_file():
    _stock_reference = pd.read_csv(os.path.join(__location__, '_stock_reference.txt'), sep='\t', encoding='latin-1')
    _stock_reference['exchange'] = _stock_reference['Exchange:Ticker'].str.split(":").str[0]
    _stock_reference['ticker'] = _stock_reference['Exchange:Ticker'].str.split(":").str[1]
    _stock_reference['ticker'] = _stock_reference['ticker'].str.upper()
    _stock_reference = _stock_reference.drop(labels=['Exchange:Ticker'], axis=1)
    _stock_reference = _stock_reference.set_index('ticker')
    for col in _stock_reference.columns.values:
        _stock_reference[col] = pd.to_numeric(_stock_reference[col], errors='coerce')
    return _stock_reference


@validate_str
def get_valuation(ticker, minority_discount=0.2):
    _key_metrics = {}
    # ensure the ticker is uppercase
    ticker = ticker.upper()
    _key_metrics['ticker'] = ticker

    # load expanded financial data from reference file
    _stock_reference = import_reference_file().loc[ticker]
    if _stock_reference.empty:
        print("Ticker Error: Ticker not in reference file.")
        return None

    # retrieve current price of the stock by ticker
    current_prices = pd.DataFrame(get_current(ticker)['Time Series (1min)']).transpose().reset_index()
    _key_metrics['current_price'] = float(
        current_prices[current_prices['index'] == current_prices['index'].max()]["4. close"].iloc[0])

    # ratios
    _key_metrics['pe_ratio'] = _stock_reference['P_LTM Diluted EPS Before Extra']
    _key_metrics['ev_ebitda'] = _stock_reference['Total Enterprise Value'] / _stock_reference['EBITDA']
    _key_metrics['ev_bv'] = _stock_reference['Total Enterprise Value'] / (_stock_reference['Total Assets'] -
                                                                          _stock_reference['Total Liabilities'])
    _key_metrics['bv_share'] = (_stock_reference['Total Assets'] -
                                _stock_reference['Total Liabilities']) / _stock_reference['Shares Outstanding']
    _key_metrics['profit_margin'] = _stock_reference['Gross Margin'] / 100
    _key_metrics['operating_margin'] = _stock_reference['Earnings from Cont Ops Margin'] / 100
    _key_metrics['roa'] = _stock_reference['Return on Assets'] / 100
    _key_metrics['roe'] = _stock_reference['Return on Equity'] / 100
    _key_metrics['debt_equity'] = _stock_reference['Total Debt_Equity'] / 100

    # graham
    _key_metrics['graham_value'] = (_stock_reference['Diluted EPS Incl Extra Items'] *
                                    (8.5 + (2 * (_stock_reference['LT EPS Growth Rate']))) * 4.4) / \
                                   ((float(get_corporate_rates()['Yield'].iloc[0].split('%')[0]) / 100) * 100)
    _key_metrics['graham_safety'] = 1 - (_key_metrics['current_price'] / _key_metrics['graham_value'])

    # Rate lookups
    _df_federal_rates = get_fed_rates()
    _fed10yr = _df_federal_rates[_df_federal_rates.Date == _df_federal_rates.Date.max()]['10yr'].iloc[0]
    _key_metrics['10yr_fed_rate'] = _fed10yr
    _inflation = get_inflation()
    _key_metrics['curr_inflation'] = _inflation

    # Intrinsic Value
    _key_metrics['book_value'] = (_stock_reference['Total Assets']-_stock_reference['Total Liabilities'])*1000000
    _key_metrics['last_annual_earnings'] = _stock_reference['Diluted EPS Incl Extra Items'] * \
        _stock_reference['Shares Outstanding']*1000000

    # 1yr growth rate
    if not pd.isnull(_stock_reference['Est Annual EPS Growth  1 Yr']):
        _key_metrics['1yr_growth_rate'] = _stock_reference['Est Annual EPS Growth  1 Yr']/100
    elif not pd.isnull(_stock_reference['LT EPS Growth Rate']):
        _key_metrics['1yr_growth_rate'] = ((_stock_reference['LT EPS Growth Rate']/100)+1)**(1/5)-1
    elif not pd.isnull(_stock_reference['Diluted EPS before Extra 10 Yr CAGR']):
        _key_metrics['1yr_growth_rate'] = _stock_reference['Diluted EPS before Extra 10 Yr CAGR']/100
    else:
        _key_metrics['1yr_growth_rate'] = 0
    # Current Year Projected Earnings
    if _key_metrics['last_annual_earnings'] > 0:
        _key_metrics['cy_proj_earn'] = _key_metrics['last_annual_earnings']*(1+_key_metrics['1yr_growth_rate'])
    else:
        _key_metrics['cy_proj_earn'] = 0
    _key_metrics['pv_cy_proj'] = _key_metrics['cy_proj_earn']/(1+_fed10yr)

    # 2yr Growth Rate
    if not pd.isnull(_stock_reference['Est Annual EPS Growth  2 Yr']):
        _key_metrics['2yr_growth_rate'] = _stock_reference['Est Annual EPS Growth  2 Yr']/100
    elif not pd.isnull(_stock_reference['LT EPS Growth Rate']):
        _key_metrics['2yr_growth_rate'] = ((_stock_reference['LT EPS Growth Rate']/100)+1)**(1/5)-1
    elif not pd.isnull(_stock_reference['Diluted EPS before Extra 10 Yr CAGR']):
        _key_metrics['2yr_growth_rate'] = _stock_reference['Diluted EPS before Extra 10 Yr CAGR']/100
    else:
        _key_metrics['2yr_growth_rate'] = 0
    # Next Year Projected Earnings
    if _key_metrics['last_annual_earnings'] > 0:
        _key_metrics['ny_proj_earn'] = _key_metrics['cy_proj_earn']*(1+_key_metrics['2yr_growth_rate'])
    else:
        _key_metrics['ny_proj_earn'] = 0
    _key_metrics['pv_ny_proj'] = _key_metrics['ny_proj_earn']/((1+_fed10yr)**2)

    # Long Term Growth Rate
    if not pd.isnull(_stock_reference['LT EPS Growth Rate']):
        _key_metrics['lt_growth_rate'] = ((_stock_reference['LT EPS Growth Rate']/100)+1)**(1/5)-1
    elif not pd.isnull(_stock_reference['Diluted EPS before Extra 10 Yr CAGR']):
        _key_metrics['lt_growth_rate'] = _stock_reference['Diluted EPS before Extra 10 Yr CAGR']/100
    else:
        _key_metrics['lt_growth_rate'] = 0
    # Present Value of Long Term Projections
    if _key_metrics['ny_proj_earn'] > 0:
        _key_metrics['pv_lt_proj'] = (_key_metrics['ny_proj_earn']*(((1+_key_metrics['lt_growth_rate'])**1) / ((1+_fed10yr)**3))) +\
                                     (_key_metrics['ny_proj_earn']*(((1+_key_metrics['lt_growth_rate'])**2) / ((1+_fed10yr)**4))) + \
                                     (_key_metrics['ny_proj_earn']*(((1+_key_metrics['lt_growth_rate'])**3) / ((1+_fed10yr)**5))) + \
                                     (_key_metrics['ny_proj_earn']*(((1+_key_metrics['lt_growth_rate'])**4) / ((1+_fed10yr)**6))) + \
                                     (_key_metrics['ny_proj_earn']*(((1+_key_metrics['lt_growth_rate'])**5) / ((1+_fed10yr)**7))) + \
                                     (_key_metrics['ny_proj_earn']*(((1+_key_metrics['lt_growth_rate'])**6) / ((1+_fed10yr)**8))) + \
                                     (_key_metrics['ny_proj_earn']*(((1+_key_metrics['lt_growth_rate'])**7) / ((1+_fed10yr)**9))) + \
                                     (_key_metrics['ny_proj_earn']*(((1+_key_metrics['lt_growth_rate'])**8) / ((1+_fed10yr)**10)))
    else:
        _key_metrics['pv_lt_proj'] = 0
    # Present value of all future earnings
    if _key_metrics['last_annual_earnings'] > 0:
        _key_metrics['pv_future_earnings'] = (_key_metrics['ny_proj_earn'] *
                                              ((((1+_key_metrics['lt_growth_rate'])**8)/((1+_fed10yr)**10)) *
                                              (((((1+_inflation)**1)/(1+_fed10yr)**1)+(((1+_inflation)**2)/(1+_fed10yr)**2) +
                                               (((1+_inflation)**3)/((1+_fed10yr)**3))+(((1+_inflation)**4)/(1+_fed10yr)**4) +
                                               (((1+_inflation)**5)/((1+_fed10yr)**5))+(((1+_inflation)**6)/(1+_fed10yr)**6) +
                                               (((1+_inflation)**7)/((1+_fed10yr)**7))+(((1+_inflation)**8)/(1+_fed10yr)**8) +
                                               (((1+_inflation)**9)/((1+_fed10yr)**9))+((1+_inflation)**10)))))
    else:
        _key_metrics['pv_future_earnings'] = 0
    _key_metrics['intrinsic_value'] = _key_metrics['book_value'] + _key_metrics['pv_cy_proj'] + \
        _key_metrics['pv_ny_proj'] + _key_metrics['pv_lt_proj'] + \
        _key_metrics['pv_future_earnings']

    _key_metrics['iv_per_share'] = _key_metrics['intrinsic_value'] / (_stock_reference['Shares Outstanding'] * 1000000)
    _key_metrics['iv_safety_margin'] = 1-(_key_metrics['current_price'] / _key_metrics['iv_per_share'])
    _key_metrics['intrinsic_fmv'] = _key_metrics['iv_per_share']*(1-minority_discount)
    # Margin of Safety at FMV
    if _key_metrics['intrinsic_fmv'] > 0:
        _key_metrics['fmv_safety_margin'] = 1-(_key_metrics['current_price']/_key_metrics['intrinsic_fmv'])
    else:
        _key_metrics['fmv_safety_margin'] = None

    return _key_metrics


def get_corporate_rates():
    _url = "http://markets.on.nytimes.com/research/markets/bonds/bonds.asp"
    _r = requests.get(_url)
    _soup = BeautifulSoup(_r.text, 'lxml')
    _table = _soup.find(class_="finra")
    _subtable = _table.find_all('tbody')[0]
    df_columns = ['Issuer', 'Cupon', 'Maturity', 'Moody', 'S&P', 'Fitch', 'Last', 'Change', 'Yield']
    df_lists = []
    for row in _subtable.find_all('tr'):
        row_list = []
        for col in row.find_all('td'):
            row_list.append(col.text)
        df_lists.append(row_list)
    df = pd.DataFrame(df_lists, columns=df_columns)
    return df


def get_fed_rates():
    _url = "https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yield"
    _r = requests.get(_url)
    _soup = BeautifulSoup(_r.text, 'lxml')
    _table = _soup.find(class_='t-chart')
    df_columns = ['Date', '1mo', '3mo', '6mo', '1yr', '2yr', '3yr', '5yr', '7yr', '10yr', '20yr', '30yr']
    df_lists = []
    # Find all the <tr> tag pairs, skip the first one, then for each.
    for row in _table.find_all('tr')[1:]:
        row_list = []
        for col in row.find_all('td'):
            row_list.append(col.string.strip())
        df_lists.append(row_list)
    df = pd.DataFrame(df_lists, columns=df_columns)
    df.Date = pd.to_datetime(df.Date)
    for col in df_columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def get_inflation():
    _url = "https://www.statbureau.org/en/united-states/inflation"
    _r = requests.get(_url)
    _soup = BeautifulSoup(_r.text, 'lxml')
    _table = _soup.find(class_="currnet-inflation-table")
    _rate_str = _table.find_all('tr')[0].find_all('td')[1].string.strip()
    return float(_rate_str.split('%')[0]) / 100


@validate_str
def get_current(ticker):
    ticker = ticker.upper()
    with open(os.path.join(__location__, 'alphavantage_api.json'), 'r') as j:
        credentials = json.load(j)
    _url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=" + ticker + \
           "&interval=1min&apikey=" + credentials['api_key']
    _r = requests.get(url=_url)
    return _r.json()


def export_valuation(ticker, location=os.getcwd(), minority_discount=0.2):
    _values = get_valuation(ticker, minority_discount)
    _column_keys = ['ticker','current_price', 'pe_ratio', 'ev_ebitda', 'ev_bv', 'bv_share', 'profit_margin',
                    'operating_margin', 'roa', 'roe', 'debt_equity', 'graham_value', 'graham_safety', 'intrinsic_fmv',
                    'fmv_safety_margin', 'book_value', 'last_annual_earnings', '1yr_growth_rate', 'cy_proj_earn',
                    'pv_cy_proj', '2yr_growth_rate', 'ny_proj_earn', 'pv_ny_proj', 'lt_growth_rate', 'pv_lt_proj',
                    'pv_future_earnings', 'intrinsic_value', 'iv_per_share', 'iv_safety_margin']
    _column_titles = ['Ticker', 'Current Price', 'P/E Ratio', 'EV/EBITDA', 'EV/BV', 'BV/Share', 'Profit Margin',
                      'Operation Margin', 'Return on Assets', 'Return on Equity', 'Debt/Equity Ratio', 'Graham Value',
                      'Graham Margin of Safety', 'Intrinsic FMV w/ Minority Discount', 'Intrinsic FMV Margin of Safety',
                      'Book Value of Equity', 'Last Annual Earnings', 'Annual Growth Rate for Current Year Projections',
                      'Current Year Projected Earnings', 'Present Value of CY Projection',
                      'Annual Growth Rate for Next Year Projections', 'Next Year Projected Earnings',
                      'Present Value of NY Projection', 'Annual Grow Rate for 3-10 Year Projections',
                      'Present Value of 3-10 Year Projections', 'Present Value of all Future Earnings',
                      'Intrinsic Value', 'Intrinsic Value per Share', 'Margin of Safety for Intrinsic Value']
    _column_data = []
    for col in _column_keys:
        _column_data.append(_values[col])
    _df = pd.DataFrame(data=[_column_data], columns=_column_titles)
    _df.to_csv(os.path.join(location, (_values["ticker"]+".csv")), index=False)
