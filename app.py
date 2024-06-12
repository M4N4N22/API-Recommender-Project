from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load data and model
data = pd.read_csv('Code_Snippets.csv')
columns_to_join = ['Code Snippet                                                                                                                                                  ', 'Description/Functionality             ','Usage Example                                                                                                    ']
data['combined_text'] = data[columns_to_join].apply(lambda row: ' '.join(row), axis=1)
embeddings = np.load('embeddings.npy')
model = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__)

def recommend_code_snippet(query, top_n=5):
    query_embedding = model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    indices = similarities.argsort()[-top_n:][::-1]
    recommendations = data.iloc[indices]
    return recommendations[['API Name          ', 'Function/Method Name      ', 'Code Snippet                                                                                                                                                  ', 'Description/Functionality             ']]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.form['query']
    recommendations = recommend_code_snippet(query)
    return render_template('results.html', query=query, recommendations=recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
