import pandas as pd
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to fix NumPy issues first
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

# Check if GPU is available in a way that won't crash
try:
    import torch
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
except Exception as e:
    print(f"Error initializing torch: {e}")
    has_gpu = False
    device = "cpu"
    print("Falling back to CPU")

def analyze_sentiment(stock_name, finbert_column='Lexrank_summary'):
    # Create the directory if it doesn't exist
    output_dir = f'sentiment_scores/{stock_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data for {stock_name}...")
    # Load the data
    df = pd.read_csv(f'processed_data/{stock_name}/{stock_name}_news_with_articles.csv')
    
    # Try to initialize the model only after all imports are done
    print("Loading FinBERT model...")
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        
        # Safely initialize the model
        if has_gpu:
            # Load with GPU support
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            # Only use half-precision if GPU is available
            model = model.to(device)
            pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
        else:
            # CPU fallback
            pipe = pipeline("text-classification", model="ProsusAI/finbert")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to simple sentiment classification...")
        # Fallback to a simpler model like TextBlob if transformers fails
        from textblob import TextBlob
        
        def simple_sentiment(text):
            if pd.isna(text) or text == "":
                return {"label": "neutral", "score": 0, "raw_score": 0.5}
            
            try:
                blob = TextBlob(str(text))
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    label = "positive"
                elif polarity < -0.1:
                    label = "negative"
                else:
                    label = "neutral"
                
                return {"label": label, "score": polarity, "raw_score": abs(polarity)}
            except:
                return {"label": "neutral", "score": 0, "raw_score": 0.5}
        
        pipe = simple_sentiment
    
    # Function to process text
    def get_sentiment(text):
        if pd.isna(text) or text == "":
            return {"label": "neutral", "score": 0, "raw_score": 0.5}

        try:
            max_tokens = 512
            
            if 'textblob' in str(pipe):
                return pipe(text)

            # Tokenize and chunk if needed
            encoded = tokenizer.encode(text, truncation=False)
            chunks = [encoded[i:i+max_tokens] for i in range(0, len(encoded), max_tokens)]

            sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
            total_score = 0
            raw_scores = []
            labels = []

            for chunk in chunks:
                decoded_chunk = tokenizer.decode(chunk, skip_special_tokens=True)
                result = pipe(decoded_chunk)[0]
                score = sentiment_map[result['label']] * result['score']
                total_score += score
                raw_scores.append(result['score'])
                labels.append(result['label'])

            avg_score = total_score / len(chunks)
            avg_raw = sum(raw_scores) / len(raw_scores)

            # Take the most frequent label among chunks
            final_label = max(set(labels), key=labels.count)

            return {
                "label": final_label,
                "score": avg_score,
                "raw_score": avg_raw
            }

        except Exception as e:
            print(f"Error processing text: {e}")
            return {"label": "neutral", "score": 0, "raw_score": 0.5}

    
    # Process articles in smaller batches to avoid memory issues
    batch_size = 10  # Start with a conservative batch size
    results = []
    
    print(f"Processing {len(df)} articles...")
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:min(i+batch_size, len(df))]
        
        for _, row in batch.iterrows():
            # Changed from Article_title to Article
            sentiment = get_sentiment(row[finbert_column]) ######## Finberted
            results.append({
                'Date': row['Date'],
                'Article_title': row['Article_title'],  # Keep this for reference
                'Stock_symbol': row['Stock_symbol'],
                'sentiment_label': sentiment['label'],
                'sentiment_score': sentiment['score'],
                'raw_score': sentiment['raw_score']
            })
    
    # Create sentiment results dataframe
    print("Creating results dataframe...")
    sentiment_df = pd.DataFrame(results)
    
    # Merge with original dataframe
    df_with_sentiment = df.merge(
        sentiment_df[['Date', 'Article_title', 'sentiment_label', 'sentiment_score', 'raw_score']],
        on=['Date', 'Article_title'],
        how='left'
    )
    
    # Calculate daily average sentiment
    daily_sentiment = sentiment_df.groupby(['Date'])['sentiment_score'].agg(['mean', 'count']).reset_index()
    daily_sentiment.columns = ['Date', 'avg_sentiment', 'article_count']
    
    # Save results to CSVs
    print(f"Saving results to {output_dir}...")
    df_with_sentiment.to_csv(f'{output_dir}/{stock_name}_with_sentiment.csv', index=False)
    daily_sentiment.to_csv(f'{output_dir}/{stock_name}_daily_sentiment.csv', index=False)
    
    # Save a separate file to indicate this is full article analysis, not just titles
    df_with_sentiment.to_csv(f'{output_dir}/{stock_name}_with_full_article_sentiment.csv', index=False)
    daily_sentiment.to_csv(f'{output_dir}/{stock_name}_daily_full_article_sentiment.csv', index=False)
    
    print(f"Full article sentiment analysis for {stock_name} completed and saved.")
    return daily_sentiment


# Example usage use finbert enviroment
if __name__ == "__main__":
    analyze_sentiment('AMZN', finbert_column='Lexrank_summary')
