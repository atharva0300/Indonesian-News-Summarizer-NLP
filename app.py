from transformers import T5ForConditionalGeneration, T5Tokenizer
from flask import Flask, render_template, request
from newspaper import Article

app = Flask(__name__)

# Load the T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

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

    # Tokenize the news content
    input_ids = tokenizer.encode("summarize: " + news_content, return_tensors="pt", max_length=512, truncation=True)

    # Generate the summary
    summary_ids = model.generate(input_ids, max_length=150, num_beams=2, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return render_template('index.html', news_content=news_content, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
