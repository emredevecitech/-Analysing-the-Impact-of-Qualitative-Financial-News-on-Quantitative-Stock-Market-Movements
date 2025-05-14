import pandas as pd
import numpy as np

def process_stock_sentiment(ticker):
    # Define file paths based on the ticker
    sentiment_path = f"sentiment_scores/{ticker}/{ticker}_daily_sentiment.csv"
    price_path = f"processed_data/{ticker}/{ticker}_price_data.csv"
    
    # Read CSV files
    sentiment_data = pd.read_csv(sentiment_path)
    price_data = pd.read_csv(price_path)

    # Rename price columns
    price_data = price_data.rename(columns={
        'date': 'Date',
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. volume': 'volume'
    })

    # Filter relevant columns
    price_filtered = price_data[["Date", "close", "volume"]]

    # Calculate returns
    price_filtered['log_return'] = np.log(price_filtered['close'] / price_filtered['close'].shift(1))
    price_filtered['simple_return'] = (price_filtered['close'] - price_filtered['close'].shift(1)) / price_filtered['close'].shift(1)

    # Convert dates to datetime.date
    sentiment_data["Date"] = pd.to_datetime(sentiment_data["Date"]).dt.date
    price_filtered["Date"] = pd.to_datetime(price_filtered["Date"]).dt.date

    # Merge dataframes
    merged_data = pd.merge(price_filtered, sentiment_data, on="Date", how="inner")

    # Save to CSV
    output_path = f'merged_data_{ticker}.csv'
    merged_data.to_csv(output_path, index=False)
    
    print(f"Saved merged data to {output_path}")

# Example usage:
process_stock_sentiment("AMZN")
