import cudf
import os
import time
import gc
import pandas as pd
from datetime import datetime

def process_stock_data(stock_symbol, 
                       start_date, 
                       end_date, 
                       news_input_file="nasdaq_exteral_data.csv", 
                       price_input_folder="Price_History/full_history", 
                       output_folder=None):
    """
    Process both news and price data for a given stock in a single function.
    
    Parameters:
    - stock_symbol: Stock ticker symbol (e.g., 'TSLA')
    - start_date: Start date in 'YYYY-MM-DD' format
    - end_date: End date in 'YYYY-MM-DD' format
    - news_input_file: CSV file containing news data
    - price_input_folder: Folder containing price history CSV files
    - output_folder: Folder to save processed data (defaults to 'processed_data/{stock_symbol}')
    
    Returns:
    - Dictionary with paths to the output files
    """
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set default output folder if not provided
    if output_folder is None:
        output_folder = f"processed_data/{stock_symbol}"
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    results = {}
    
    # Step 1: Process news data
    print(f"Processing news data for {stock_symbol}...")
    news_results = _process_news_data(stock_symbol, news_input_file, output_folder)
    results.update(news_results)
    
    # Step http://2:.Process price data
    print(f"Processing price data for {stock_symbol} from {start_date} to {end_date}...")
    price_file = _process_price_data(stock_symbol, start_date, end_date, price_input_folder, output_folder)
    results['price_data'] = price_file
    
    print(f"All processing completed in {time.time() - start_time:.2f} seconds")
    print(f"Results saved to: {output_folder}")
    
    return results

def _process_news_data(stock_symbol, input_file, output_folder, chunk_size=100000):
    """Process news data for the specified stock symbol"""
    results = {}
    
    # Only read necessary columns to avoid memory issues
    usecols = ['Date', 'Stock_symbol', 'Article_title', 'Url', 
               'Lsa_summary', 'Luhn_summary', 'Textrank_summary', 'Lexrank_summary']
    
    # Try using cuDF for faster processing (GPU-accelerated)
    try:
        # Option 1: Use cuDF if data fits in GPU memory
        df = cudf.read_csv(input_file, usecols=usecols)
        filtered_df = df[df['Stock_symbol'] == stock_symbol]
        filtered_df = filtered_df.sort_values('Date')
        
        # Save intermediate filtered data
        filtered_file = os.path.join(output_folder, f"{stock_symbol}_all_news.csv")
        filtered_df.to_csv(filtered_file, index=False)
        results['all_news'] = filtered_file
        
        # Convert to pandas for processing
        pdf = filtered_df.to_pandas()
        
    except (MemoryError, RuntimeError):
        # Option 2: Fall back to pandas chunking if GPU memory is exceeded
        print("GPU memory exceeded, falling back to pandas chunk processing")
        
        chunks = []
        for chunk in pd.read_csv(input_file, chunksize=chunk_size, usecols=usecols):
            filtered = chunk[chunk['Stock_symbol'] == stock_symbol]
            if len(filtered) > 0:
                chunks.append(filtered)
        
        if chunks:
            pdf = pd.concat(chunks, ignore_index=True)
            pdf = pdf.sort_values('Date')
            
            # Save intermediate filtered data
            filtered_file = os.path.join(output_folder, f"{stock_symbol}_all_news.csv")
            pdf.to_csv(filtered_file, index=False)
            results['all_news'] = filtered_file
        else:
            print(f"No news records found for {stock_symbol}")
            return results
    
    # Split data into two DataFrames: one with articles and one without
    df_with_articles = pdf.dropna(subset=['Lsa_summary'])
    df_without_articles = pdf[pdf['Lsa_summary'].isna()]
    
    # Save the cleaned DataFrames
    with_articles_file = os.path.join(output_folder, f"{stock_symbol}_news_with_articles.csv")
    without_articles_file = os.path.join(output_folder, f"{stock_symbol}_news_without_articles.csv")
    
    df_with_articles.to_csv(with_articles_file, index=False)
    df_without_articles.to_csv(without_articles_file, index=False)
    
    results['news_with_articles'] = with_articles_file
    results['news_without_articles'] = without_articles_file
    
    print(f"Processed {len(pdf)} news records for {stock_symbol}")
    print(f"  - {len(df_with_articles)} records with article summaries")
    print(f"  - {len(df_without_articles)} records without article summaries")
    
    return results

def _process_price_data(stock_symbol, start_date, end_date, input_folder, output_folder):
    """Process price data for the specified stock symbol and date range"""
    # Load the stock data
    file_path = os.path.join(input_folder, f'{stock_symbol}.csv')
    
    try:
        stock_data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Price data file for {stock_symbol} not found at {file_path}")
        return None
    
    # Ensure the 'Date' column is in datetime format
    date_col = 'date' if 'date' in stock_data.columns else 'Date'
    stock_data[date_col] = pd.to_datetime(stock_data[date_col])
    
    # Standardize column name
    if date_col != 'Date':
        stock_data.rename(columns={date_col: 'Date'}, inplace=True)
    
    # Filter data by date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    mask = (stock_data['Date'] >= start_dt) & (stock_data['Date'] <= end_dt)
    filtered_data = stock_data[mask]
    
    # Sort data by date
    filtered_data = filtered_data.sort_values(by='Date')
    
    # Save filtered data to CSV
    output_file = os.path.join(output_folder, f'{stock_symbol}_price_data.csv')
    filtered_data.to_csv(output_file, index=False)
    
    print(f"Processed {len(filtered_data)} price records for {stock_symbol} from {start_date} to {end_date}")
    
    return output_file

def clear_memory():
    """Clear memory to prevent out-of-memory errors"""
    gc.collect()
    try:
        # Only run if running in a CUDA environment
        from numba import cuda
        cuda.get_current_device().reset()
        print("CUDA memory cleared")
    except (ImportError, ModuleNotFoundError):
        pass

if __name__ == "__main__":
    # Example usage with command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Process stock data')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol (e.g., TSLA)')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--news-file', type=str, default="nasdaq_exteral_data.csv", help='News data CSV file')
    parser.add_argument('--price-folder', type=str, default="Price_History/full_history", help='Price history folder')
    parser.add_argument('--output', type=str, default=None, help='Output folder')
    
    args = parser.parse_args()
    
    try:
        results = process_stock_data(
            args.symbol, 
            args.start, 
            args.end, 
            args.news_file, 
            args.price_folder, 
            args.output
        )
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        clear_memory()

# Example usage without command line:
# results = process_stock_data("TSLA", "2010-01-01", "2024-02-01")
# python prepare_financial_data.py --symbol AMZN --start 2010-01-01 --end 2024-02-01
