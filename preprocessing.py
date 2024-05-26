# Case folding
def case_folding(question_text):
    question_text = str.lower(question_text) 
    return question_text

# Cleansing
import re
def cleansing(question_text):
    question_text = re.sub(r'https?://\S+', ' ', question_text) #remove link
    question_text = re.sub(r'\d+', ' ', question_text) #remove digit
    question_text = re.sub(r'[^\w\s]', ' ', question_text) #remove punctuation, symbol
    return question_text

# Tokenizing
def tokenizing(question_text):
    question_text = question_text.split()
    return question_text

# Stemmming
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

def stemming(question_text):
    stemmer = PorterStemmer()
    question_text = [stemmer.stem(word) for word in question_text]
    return question_text

def lemmatizing(question_text):
    lemmatizer = WordNetLemmatizer()
    question_text = [lemmatizer.lemmatize(word) for word in question_text]
    return question_text

# Stop words
custom_stopwords = ['does', 'was', 'would', 'one', 'two', 'three', 'four', 'five', 'six', 'nine', 'ten']
def stopword_remove(question_text):
    stop_words = stopwords.words('english')
    stop_words.extend(custom_stopwords)
    question_text = [word for word in question_text if word not in stop_words]
    return question_text

# Combine all preprocessing
def preprocessing_text_with_stemming(question_text):
    question_text = case_folding(question_text)
    question_text = cleansing(question_text)
    question_text = tokenizing(question_text)
    question_text = stemming(question_text)
    question_text = stopword_remove(question_text)
    return question_text

def preprocessing_text_with_lemma(question_text):
    question_text = case_folding(question_text)
    question_text = cleansing(question_text)
    question_text = tokenizing(question_text)
    question_text = stopword_remove(question_text)
    question_text = lemmatizing(question_text)
    return question_text