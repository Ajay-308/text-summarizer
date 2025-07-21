# 📝 Text Summarizer and Translator

**Live Demo:** [text-sumarize.streamlit.app](https://text-sumarize.streamlit.app/)  
**Author:** [Ajay-308](https://github.com/Ajay-308)

A simple NLP-powered Streamlit web app that summarizes English text and translates the summary into other languages. It uses GloVe word embeddings for sentence vectorization and a graph-based ranking algorithm for extractive summarization.

---

## 🚀 Features

- 🔤 **Text Summarization** using cosine similarity & PageRank
- 🌐 **Language Detection** with `langdetect`
- 🌍 **Translation** using `deep-translator`
- 📚 Built using `nltk`, `scikit-learn`, `networkx`, and `Streamlit`
- ✨ Embedding powered by `glove.6B.100d.txt`

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ajay-308/text-summarizer.git
   cd text-summarizer
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Download GloVe embeddings

Download glove.6B.100d.txt from https://nlp.stanford.edu/data/glove.6B.zip

Extract and place glove.6B.100d.txt in the root directory of this project.

Run the Streamlit app

bash
Copy
Edit
streamlit run web.py
🧠 How It Works
Text Input – The user enters a paragraph or article.

Preprocessing – The app lowercases the text, removes stopwords and punctuation.

Sentence Embeddings – Uses GloVe vectors to create sentence representations.

Similarity Matrix – Calculates cosine similarity between sentence vectors.

Graph Ranking – Builds a similarity graph and ranks sentences using PageRank.

Translation – Translates top-ranked summary sentences into the selected language.

🔧 Tech Stack
Frontend: Streamlit

NLP: NLTK, Scikit-learn, NetworkX

Translation: Deep Translator (deep-translator)

Embeddings: GloVe 100D word vectors

📁 Requirements
txt
Copy
Edit
nltk==3.8.1
streamlit==1.33.0
networkx==3.3
langdetect==1.0.9
deep-translator==1.11.4
numpy
scikit-learn
📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

📌 Topics
nltk · sklearn · streamlit · text-summarization · word-embeddings · translation · nlp · graph-ranking
