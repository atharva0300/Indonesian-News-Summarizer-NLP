from transformers import T5ForConditionalGeneration, T5Tokenizer, BartTokenizer, BartForConditionalGeneration
from flask import Flask, render_template, request
from newspaper import Article
from rouge import Rouge

app = Flask(__name__, static_url_path='/static')

# Load the T5 model and tokenizer
model_t5 = T5ForConditionalGeneration.from_pretrained('D:\GitHub\Indonesian-News-Summarizer-NLP\Outputs\\t5_outputs')
tokenizer_t5 = T5Tokenizer.from_pretrained('D:\GitHub\Indonesian-News-Summarizer-NLP\Outputs\\t5_outputs')

# Load the BART model and tokenizer
model_bart = BartForConditionalGeneration.from_pretrained("D:\GitHub\Indonesian-News-Summarizer-NLP\Outputs\\bart_outputs\checkpoint-8000-20240416T045815Z-002\checkpoint-8000")
tokenizer_bart = BartTokenizer.from_pretrained("D:\GitHub\Indonesian-News-Summarizer-NLP\Outputs\\bart_outputs\checkpoint-8000-20240416T045815Z-002\checkpoint-8000")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    # Get the news content from the form
    news_url = request.form['news_url']

    # Extract the article content from the news URL
    article = Article(news_url)
    article.download()
    article.parse()
    news_content = article.text
    
    # Tokenize the news content for T5
    input_ids_t5 = tokenizer_t5.encode("summarize: " + news_content, return_tensors="pt", max_length=512, truncation=True)

    # Generate the T5 summary
    summary_ids_t5 = model_t5.generate(input_ids_t5, max_length=500, num_beams=2, early_stopping=True)
    summary_t5 = tokenizer_t5.decode(summary_ids_t5[0], skip_special_tokens=True)

    # Tokenize the news content for BART
    input_ids_bart = tokenizer_bart.encode(news_content, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the BART summary
    summary_ids_bart = model_bart.generate(input_ids_bart, max_length=500, num_beams=2, early_stopping=True)
    summary_bart = tokenizer_bart.decode(summary_ids_bart[0], skip_special_tokens=True)

    # Calculate word count for news content and summaries
    word_count_content = len(news_content.split())
    word_count_t5 = len(summary_t5.split())
    word_count_bart = len(summary_bart.split())

    # Calculate ROUGE scores for T5
    rouge = Rouge()
    rouge_scores_t5 = rouge.get_scores(summary_t5, news_content, avg=True)

    # Calculate ROUGE scores for BART
    rouge_scores_bart = rouge.get_scores(summary_bart, news_content, avg=True)

    return render_template('index.html', 
                           news_content=news_content, 
                           summary_t5=summary_t5, 
                           summary_bart=summary_bart,
                           word_count_content=word_count_content, 
                           word_count_t5=word_count_t5, 
                           word_count_bart=word_count_bart,
                           rouge_scores_t5=rouge_scores_t5,
                           rouge_scores_bart=rouge_scores_bart
                        )

if __name__ == '__main__':
    app.run(debug=True)
