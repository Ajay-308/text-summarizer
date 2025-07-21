import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from langdetect import detect_langs
from deep_translator import GoogleTranslator

nltk.download('punkt')
nltk.download('stopwords')

# Load GloVe word embeddings
word_embeddings = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs

# Preprocessing functions
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    return ' '.join([w for w in word_tokens if w.lower() not in stop_words])

def remove_punctuations(text):
    translator = str.maketrans('', '', punctuation)
    return text.translate(translator)

def convert_lower(text):
    return text.lower()

# Text summarization
def generate_summary(text, word_embeddings):
    lower_text = convert_lower(text)
    no_stop = remove_stop_words(lower_text)
    no_punct = remove_punctuations(no_stop)
    sentences = sent_tokenize(text)

    sentence_vectors = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        if len(words) > 0:
            vec = np.mean([word_embeddings.get(word.lower(), np.zeros((100,))) for word in words], axis=0)
        else:
            vec = np.zeros((100,))
        sentence_vectors.append(vec)

    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, -1), sentence_vectors[j].reshape(1, -1))[0, 0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    return ranked_sentences

# Streamlit UI
def main():
    st.title("Text Summarization and Translation App")
    text = st.text_area("Enter your text here:")

    lang_options = {
        'en': 'English', 'hi': 'Hindi', 'fr': 'French', 'es': 'Spanish', 'de': 'German',
        'ru': 'Russian', 'ja': 'Japanese', 'zh-cn': 'Chinese (Simplified)', 'ko': 'Korean',
        'ar': 'Arabic', 'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu', 'tr': 'Turkish',
        'it': 'Italian', 'pt': 'Portuguese', 'ur': 'Urdu', 'vi': 'Vietnamese'
    }

    selected_lang = st.selectbox("Select language for translation:", list(lang_options.values()))

    if st.button("Summarize"):
        if not text.strip():
            st.warning("Please enter some text to summarize.")
            return

        # Generate summary
        summary = generate_summary(text, word_embeddings)

        st.subheader("Original Text:")
        st.write(text)

        st.subheader("Summarized Text:")
        for i in range(min(5, len(summary))):
            st.write(summary[i][1])

        # Detect language
        detected_lang = detect_langs(text)
        st.write("Detected Language(s):", detected_lang)

        # Translate summary
        lang_code = [k for k, v in lang_options.items() if v == selected_lang][0]
        st.subheader("Translated Summary:")
        for i in range(min(5, len(summary))):
            sentence = summary[i][1]
            try:
                translated = GoogleTranslator(source='auto', target=lang_code).translate(sentence)
                st.write(translated)
            except Exception as e:
                st.error(f"Error translating sentence: {e}")

if __name__ == "__main__":
    main()
