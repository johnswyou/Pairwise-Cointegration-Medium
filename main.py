import requests
import pandas as pd
import time
import os
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt

# -------------------
# DATA BULK DOWNLOAD
# -------------------

# Dependencies:
# - stock_tickers.txt

api_key = "YOUR_API_KEY"

def av_api_call(ticker, function='TIME_SERIES_DAILY_ADJUSTED'):
    url = f'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={api_key}&outputsize=full'
    r = requests.get(url)
    data = r.json()
    return data

with open('stock_tickers.txt') as file:
    stock_tickers = [line.rstrip() for line in file]

counter = 0

# NOTE: This will take a while to run
# NOTE: The API limits you to 5 calls per minute
# NOTE: The API limits you to 500 calls per day (for free accounts)
# NOTE: bulk_download folder must exist

for ticker in stock_tickers:
    print(f"Downloading ticker: {ticker}")
    try:
        data = av_api_call(ticker)
        current_ticker_data = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient="index").reset_index(names = 'date')
        current_ticker_data["ticker"] = ticker
        current_ticker_data.to_csv('bulk_download/' + ticker + '.csv', index=False)

        counter += 1
        if counter % 5 == 0:
            print("Sleeping for 61 seconds...")
            time.sleep(61)
    except:
        print(f"Could not get ticker: {ticker}")
        pass

# -------------------
# GET ADJUSTED CLOSES
# -------------------

start_date = "1900-01-01"
end_date = "2100-12-31"

# Initialize an empty dataframe to store the closing prices
all_closes = pd.DataFrame()

# Loop through all CSV files in the directory and extract the closing prices
for filename in tqdm(os.listdir("bulk_download")):
    if filename.endswith(".csv"):
        # Extract the ticker from the filename
        ticker = filename.split(".")[0]
        
        # Read the CSV file into a pandas dataframe
        df = pd.read_csv(os.path.join("bulk_download", filename))
        
        # Filter the dataframe by date range
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        
        # Select the closing prices and rename the column
        closes = df[["date", "5. adjusted close"]]
        closes.columns = ["date", ticker]
        closes = closes.set_index("date")
        
        # Add the closing prices to the all_closes dataframe
        all_closes = pd.concat([all_closes, closes], axis=1)

all_closes_names = list(all_closes)

with open("all_closes_names.pkl", "wb") as f:
    pickle.dump(all_closes_names, f)

with open("all_closes_df.pkl", "wb") as f:
    pickle.dump(all_closes, f)

# ---------------------
# COMPUTE CORRELATIONS
# ---------------------

def filter_dataframe_by_date_range(df, start_date, end_date):
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.sort_index()
    filtered_df = df.loc[start_date:end_date]
    if filtered_df.empty:
        # if no rows found in specified date range, filter based on next available date key
        filtered_df = df.loc[df.index[df.index >= start_date][0]:df.index[df.index <= end_date][-1]]
    return filtered_df

# with open("all_closes_df.pkl", "rb") as f:
#     all_closes_df = pickle.load(f)

start_date = '2003-03-30' # Specify the start date
end_date = '2023-03-30' # Specify the end date
subset_all_closes_df = filter_dataframe_by_date_range(all_closes, start_date, end_date)
subset_all_closes_df = subset_all_closes_df.dropna(axis=1)

# Compute the correlation matrix
corr = subset_all_closes_df.corr()

def extract_upper_triangle(corr_df):
    """
    Takes the output of DataFrame.corr() and returns a new data frame containing
    only the upper triangular elements of the correlation matrix, along with the
    corresponding column and row names.
    """
    # Get the upper triangle of the correlation matrix
    upper_tri = np.triu(corr_df, k=1)

    # Create a list of column and row names for the upper triangle elements
    col_names, row_names = np.where(upper_tri)
    col_names = corr_df.columns[col_names]
    row_names = corr_df.index[row_names]

    # Create a new data frame with the upper triangle values and column/row names
    upper_tri_df = pd.DataFrame({'Column': col_names, 'Row': row_names, 'Correlation': upper_tri[upper_tri != 0]})

    return upper_tri_df

# Extract the upper triangle of the correlation matrix
corr_df = extract_upper_triangle(corr).sort_values(by='Correlation', ascending=False)

# Filter for highly correlated pairs
def get_pairs(df):
    pairs = []
    for index, row in df.iterrows():
        pairs.append((row['Column'], row['Row']))
    return pairs

def pair_cointegration(ticker1, ticker2, df):

    _, pvalueA, _ = coint(df.loc[:,ticker1], df.loc[:,ticker2])
    _, pvalueB, _ = coint(df.loc[:,ticker2], df.loc[:,ticker1])

    if pvalueA < pvalueB:
        return pvalueA
    else:
        return pvalueB
    
highly_correlated_pairs = get_pairs(corr_df[corr_df['Correlation'] > 0.99])

# ----------------------------
# COMPUTE COINTEGRATION TESTS
# ----------------------------

# Compute the CADF test p-values for each highly correlated pair
coint_pairs = {}

for pair in tqdm(highly_correlated_pairs):
    ticker1 = pair[0]
    ticker2 = pair[1]
    dict_key = ticker1 + '-' + ticker2
    pvalue = pair_cointegration(ticker1, ticker2, subset_all_closes_df)
    coint_pairs[dict_key] = pvalue

coint_pairs_df = pd.DataFrame.from_dict(coint_pairs, orient='index')
coint_pairs_df.columns = ['pvalue']

# Filter for pairs with a p-value less than 0.05
potentially_cointegrated_pairs = coint_pairs_df[coint_pairs_df['pvalue'] < 0.05].sort_values('pvalue')

# -----------------
# PLOT TIME SERIES
# -----------------

# Here's a utility function to plot each pair's time series
def plot_time_series(all_closes_df, col1, col2):
    # Select the two columns of interest
    data = all_closes_df[[col1, col2]]

    # Plot the data
    data.plot(figsize=(10, 6))
    plt.yscale('log')

    # Set the axis labels and title
    plt.xlabel('Date')
    plt.ylabel('Adjusted closing price')
    plt.title(f'{col1} and {col2}')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

# Example usage
# plot_time_series(subset_all_closes_df.sort_index(), "MQT", "MYD")