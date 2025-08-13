# Text-Analysis
A Simple Flask based Web Application that performs summarization, sentiment analysis and Keyword extraction 
## Features
- **Summarization** - Uses `facebook/bart-large-cnn` for generating concise summaries.
- **Sentiment Analysis** -Uses `distilbert-base-uncases-finetunes-sst-2-english` to determine text sentiment.
- **Keyword Extraction** - Uses RAKE for key phrase identification.
- **Combined Analysis** - Runs all three analyses at once.
- **History Tracking** - Stores the last 5 Processed inputs and outputs.
