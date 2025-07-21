📝 Text Summarizer and Translator
Live Demo: text-sumarize.streamlit.app
Author: Ajay-308

This project is a lightweight NLP-powered Streamlit web app that summarizes English text and translates the summary into multiple languages. It combines sentence ranking (TextRank) with GloVe word embeddings for extractive summarization and supports over 50 translation languages via deep-translator.

🚀 Features
🔤 Text Summarization using cosine similarity & PageRank

🌐 Language Detection with langdetect

🌍 Translation of summaries using deep-translator

📚 Built with nltk, scikit-learn, networkx, and Streamlit

✨ Supports GloVe word embeddings (glove.6B.100d.txt)

📦 Requirements
makefile
Copy
Edit
nltk==3.8.1
streamlit==1.33.0
networkx==3.3
langdetect==1.0.9
deep-translator==1.11.4
numpy
scikit-learn
🧠 How It Works
Input Text → User enters a paragraph.

Preprocessing → Converts to lowercase, removes stopwords and punctuation.

Vectorization → Computes sentence vectors using GloVe.

Similarity Matrix → Builds sentence graph using cosine similarity.

Ranking → Applies PageRank to identify the most important sentences.

Translation → Top-ranked sentences are translated into the selected language.

🖥️ Run Locally
bash
Copy
Edit
git clone https://github.com/Ajay-308/text-summarizer.git
cd text-summarizer
pip install -r requirements.txt
streamlit run web.py
Make sure to place glove.6B.100d.txt in the project directory.

📌 Topics & Tags
nltk · sklearn · streamlit · vectorizer · word-embeddings · langdetect · translation · text-summarization

📄 License
This project is open source and available under the MIT License (you can add one if you want).

