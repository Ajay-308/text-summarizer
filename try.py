import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from langdetect import detect_langs
from googletrans import Translator

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

# Function to remove stopwords
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    filtered_text = ' '.join([w for w in word_tokens if not w in stop_words])
    return filtered_text

# Function to remove punctuations
def remove_punctuations(text):
    translator = str.maketrans('', '', punctuation)
    text_without_punctuations = text.translate(translator)
    sentences = sent_tokenize(text_without_punctuations)
    return sentences

# Function to convert text to lowercase
def convert_lower(text):
    return text.lower()

# Function to generate summary
def generate_summary(text, word_embeddings):
    lower_text = convert_lower(text)
    new_text = remove_stop_words(lower_text)
    sentences = sent_tokenize(text)
    cleaned_sentences = remove_punctuations(new_text)

    sentence_vectors = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        sentence_vector = np.mean([word_embeddings.get(word, np.zeros((100,))) for word in words], axis=0)
        sentence_vectors.append(sentence_vector)

    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, -1), sentence_vectors[j].reshape(1, -1))[0, 0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    return ranked_sentences

# Streamlit UI
def main():
    st.title("Text Summarization App")
    text = st.text_area("Enter your text here:")
    
    lang_options = {
        'af': 'Afrikaans', 'sq': 'Albanian', 'am': 'Amharic', 'ar': 'Arabic', 'hy': 'Armenian', 'az': 'Azerbaijani',
        'eu': 'Basque', 'be': 'Belarusian', 'bn': 'Bengali', 'bs': 'Bosnian', 'bg': 'Bulgarian', 'ca': 'Catalan',
        'ceb': 'Cebuano', 'ny': 'Chichewa', 'zh-cn': 'Chinese (Simplified)', 'zh-tw': 'Chinese (Traditional)',
        'co': 'Corsican', 'hr': 'Croatian', 'cs': 'Czech', 'da': 'Danish', 'nl': 'Dutch', 'en': 'English',
        'eo': 'Esperanto', 'et': 'Estonian', 'tl': 'Filipino', 'fi': 'Finnish', 'fr': 'French', 'fy': 'Frisian',
        'gl': 'Galician', 'ka': 'Georgian', 'de': 'German', 'el': 'Greek', 'gu': 'Gujarati', 'ht': 'Haitian Creole',
        'ha': 'Hausa', 'haw': 'Hawaiian', 'iw': 'Hebrew', 'he': 'Hebrew', 'hi': 'Hindi', 'hmn': 'Hmong',
        'hu': 'Hungarian', 'is': 'Icelandic', 'ig': 'Igbo', 'id': 'Indonesian', 'ga': 'Irish', 'it': 'Italian',
        'ja': 'Japanese', 'jw': 'Javanese', 'kn': 'Kannada', 'kk': 'Kazakh', 'km': 'Khmer', 'ko': 'Korean',
        'ku': 'Kurdish (Kurmanji)', 'ky': 'Kyrgyz', 'lo': 'Lao', 'la': 'Latin', 'lv': 'Latvian', 'lt': 'Lithuanian',
        'lb': 'Luxembourgish', 'mk': 'Macedonian', 'mg': 'Malagasy', 'ms': 'Malay', 'ml': 'Malayalam',
        'mt': 'Maltese', 'mi': 'Maori', 'mr': 'Marathi', 'mn': 'Mongolian', 'my': 'Myanmar (Burmese)', 'ne': 'Nepali',
        'no': 'Norwegian', 'or': 'Odia', 'ps': 'Pashto', 'fa': 'Persian', 'pl': 'Polish', 'pt': 'Portuguese',
        'pa': 'Punjabi', 'ro': 'Romanian', 'ru': 'Russian', 'sm': 'Samoan', 'gd': 'Scots Gaelic', 'sr': 'Serbian',
        'st': 'Sesotho', 'sn': 'Shona', 'sd': 'Sindhi', 'si': 'Sinhala', 'sk': 'Slovak', 'sl': 'Slovenian',
        'so': 'Somali', 'es': 'Spanish', 'su': 'Sundanese', 'sw': 'Swahili', 'sv': 'Swedish', 'tg': 'Tajik',
        'ta': 'Tamil', 'te': 'Telugu', 'th': 'Thai', 'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu', 'ug': 'Uyghur',
        'uz': 'Uzbek', 'vi': 'Vietnamese', 'cy': 'Welsh', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'zu': 'Zulu'
    }
    
    trans_lang = st.selectbox("Select language for translation:", list(lang_options.values()))

    if st.button("Summarize"):
        if text.strip() == "":
            st.warning("Please enter some text to summarize.")
        else:
            lang_code = [k for k, v in lang_options.items() if v == trans_lang][0]
            summarized_text = generate_summary(text, word_embeddings)

            st.subheader("Original Text:")
            st.write(text)

            st.subheader("Summarized Text:")
            for i in range(min(5, len(summarized_text))):
                st.write(summarized_text[i][1])


            # Language detection
            lang_detection = detect_langs(text)
            st.write("Detected language:", lang_detection)

            # Translation
            translator = Translator()
            translated_sentences = []
            for _, sentence in summarized_text[:5]:
                try:
                    translated_sentence = translator.translate(sentence, dest=lang_code).text
                    translated_sentences.append(translated_sentence)
                except Exception as e:
                    st.error(f"Translation failed for '{sentence}' with error: {e}")

            st.subheader("Translated Summary:")
            for translated_sentence in translated_sentences:
                st.write(translated_sentence)

if __name__ == "__main__":
    main()
