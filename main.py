from flask import Flask, render_template, request, jsonify, redirect, url_for
from exa_py import Exa
from openai import OpenAI
import sqlite3
import sqlite_vec
import re
import struct
from typing import List
import json
import os

app = Flask(__name__)

exa = Exa(os.getenv('EXA_API_KEY'))
client = OpenAI(os.getenv('OPENAI_API_KEY'))


def serialize(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)


websites_dictionary = {
    "bloomberg.com": "Bloomberg",
    "prnewswire.com": "PR Newswire",
    "marketwatch.com": "MarketWatch",
    "axios.com": "Axios",
    "apnews.com": "AP",
    "wsj.com": "Wall Street Journal",
    "theinformation.com": "Information",
    "bostonglobe.com": "Boston Globe",
    "washingtonpost.com": "Washington Post",
    "economist.com": "Economist",
    "ft.com": "Financial Times",
    "reuters.com": "Reuters",
    "nytimes.com": "New York Times",
    "fortune.com": "Fortune",
    "techcrunch.com": "TechCrunch",
    "venturebeat.com": "VentureBeat"
}


def extract_publisher(url):
    pattern = r"(?:https?://)?(?:www\.)?([A-Za-z0-9_-]+)(?:\.[A-Za-z]+)+"
    match = re.search(pattern, url)
    if match:
        domain = match.group(1)
        full_domain = f"{domain}.com"
        return websites_dictionary.get(full_domain, domain)
    return None


@app.route('/welcome')
def landing():
    return render_template('landing.html')


@app.route('/submit_email', methods=['POST'])
def submit_email():
    email = request.form['email']

    connection = sqlite3.connect("emails.db")
    cursor = connection.cursor()

    cursor.execute("INSERT INTO emails (email) VALUES (?)", (email, ))
    connection.commit()
    connection.close()

    return redirect(url_for('index'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search_news():
    query = request.form['query']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    exa_results = exa.search_and_contents(
        query,
        type="neural",
        use_autoprompt=True,
        num_results=20,
        text=True,
        start_published_date=start_date,
        end_published_date=end_date,
        category="news",
        include_domains=[
            "bloomberg.com", "wsj.com", "theinformation.com",
            "bostonglobe.com", "washingtonpost.com", "economist.com", "ft.com",
            "reuters.com", "nytimes.com", "fortune.com", "axios.com",
            "apnews.com", "prnewswire.com", "marketwatch.com",
            "techcrunch.com", "venturebeat.com"
        ])

    articles = []
    for item in exa_results.results:
        publisher = extract_publisher(item.url)
        articles.append({
            'title': item.title,
            'published_date': item.published_date,
            'source': item.url,
            'publisher': publisher,
            'content': item.text[:300] + '...',
            'author': item.author,
        })

    return jsonify({'articles': articles})


@app.route('/analyze', methods=['POST'])
def analyze_headline():
    headline = request.json['headline']
    embedding_results = search_embedding(headline)
    return jsonify({'embedding_results': embedding_results})


def search_embedding(query):
    db = sqlite3.connect("companies.db")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    query_embedding = client.embeddings.create(
        input=query, model="text-embedding-3-small").data[0].embedding

    results = db.execute(
        """
        SELECT
          vec_chunks_quantized.id,
          distance,
          company,
          date,
          chunk
        FROM vec_chunks_quantized
        LEFT JOIN chunks ON chunks.id = vec_chunks_quantized.id
        WHERE chunk_embedding_coarse MATCH vec_quantize_binary(?)
          AND k = 20
        ORDER BY distance
        """,
        [json.dumps(query_embedding)],
    ).fetchall()

    db.close()

    processed_results = [{
        'id': row[0],
        'distance': row[1],
        'company': row[2],
        'date': row[3],
        'chunk': row[4],
    } for row in results]

    return processed_results


if __name__ == '__main__':
    app.run()
