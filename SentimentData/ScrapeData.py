from bs4 import BeautifulSoup
import requests
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from functools import lru_cache         

stocks = ["AMD", "NVDA", "INTC", "MSFT"]
merged_df = pd.DataFrame()

# We call the API just once, to avoid call limits 
@lru_cache(maxsize=1)         
def get_classifier():
    tok   = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return pipeline("text-classification", model=model, tokenizer=tok, top_k=1)

def run_sentiment(headlines):
    clf = get_classifier()
    return clf(headlines, batch_size=32, truncation=True)   

def scrape_news_articles():
        
    columns = ['datetime','title','top_sentiment','sentiment_score']
    df = pd.DataFrame(columns = columns)
    for page in range(1,44):
            
        url = f'https://markets.businessinsider.com/news/tsm-stock?p={page}'
        response = requests.get(url)
        html = response.text
        soup = BeautifulSoup(html,'lxml')
        articles = soup.find_all('div', class_ = 'latest-news__story')

            
        for article in articles:
            datetime = article.find('time', class_ ='latest-news__date').get('datetime')
            title = article.find('a', class_ ='news-link').text

            top_sentiment = ''
            sentiment_score = 0
            df = pd.concat([pd.DataFrame([[datetime,title,top_sentiment,sentiment_score]], columns = df.columns),df],ignore_index= True)
            
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['date'] = df['datetime'].dt.date


    # Filter within date range
    start_date = pd.to_datetime("2018-01-03")
    end_date = pd.to_datetime("2025-02-27")

    df_filtered = df[(df['date'] >= start_date.date()) & (df['date'] <= end_date.date())]
    df_filtered.index = pd.to_datetime(df_filtered['date'])
    df_filtered.drop(columns = ['datetime','date'],inplace = True)
    
    return df_filtered


df = scrape_news_articles()
titles= df["title"].tolist()
results  = run_sentiment( titles)         

# Results is al list of dicitionaries which need to be unwrapped
flat = [r[0] for r in results]
df["top_sentiment"]   = [d["label"].lower() for d in flat]
df["sentiment_score"] = [d["score"] for d in flat]

# Map sentiment ratings to categorical variables
label2num = {"negative": 0, "neutral": 1, "positive": 2}
df["top_sentiment"] = df["top_sentiment"].map(label2num)

def majority_sentiment(group):
    #Perform a majority vote. If we have more than one article per day, we pick the sentiment score that occurs most often.

    vc = group["top_sentiment"].value_counts()
    if vc.iloc[0] > 1 or len(vc) == 1:         
        return vc.idxmax()
    # If there are equally many articles with the same rating, we pick one with the highest softmax probability.
    return  group.sort_values("sentiment_score", ascending=False)["top_sentiment"].iloc[0]  

#Index the same as log returns and kernel observations
df.index = df.index.normalize()
daily_sent = df.groupby(df.index).apply(majority_sentiment)


daily_sent = daily_sent.reindex(merged_df.index)
daily_sent.to_csv(r"C:\Users\rickt\Desktop\Econometrics and Data Science\Thesis\Data\Stock\High Frequency\TSM\TSM_daily_sent.csv", index=True)

