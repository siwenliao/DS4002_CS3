# Goal: Clean and standardize the data to prepare for LDA
# Steps: import and understand the data, clean text by removing punctuation and digits, remove stopwords, and perform lemmatization (reducing words down to their base form).

import spacy.cli
spacy.cli.download("en_core_web_md")
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
import en_core_web_md
import re
nltk.download('wordnet')
nltk.download('stopwords')

df = pd.read_csv('/content/DS-4002_Group5_Project1/DATA/McDonalds_Reviews.csv', encoding="latin1")
nlp = en_core_web_md.load(disable=['parser', 'ner'])

# Subsetting our data to only variables of interest
df = df[['review','rating']]

# Understanding the data
print(df.head(), '\n')
print(df.tail(), '\n')
print('Number of reviews:', len(df), '\n')
print('Checking for missing values: \n', df.isnull().sum(), '\n')

# An extensive list of English stopwords
 # (commonly-used, high-frequency words that carry little semantic meaning on their own)
stop_words = set(stopwords.words('english'))

# Cleaning text
def clean_text(text):

  # Lowercase
  text = str(text).lower()

  # Remove puncuation
  delete_dict = {punc: '' for punc in string.punctuation}
  table = str.maketrans(delete_dict)
  text_nopunc = text.translate(table)

  text_nopunc = re.sub(r"[^a-zA-Z0-9\s]", "", text_nopunc)

  # Split into a list (tokens)
  text_arr = text_nopunc.split()

  # Remove digits, one-letter words, and stopwords
  text_keep = ' '.join([w for w in text_arr if (not w.isdigit())
                and (len(w) > 1)
                and (w not in stop_words)])

  return text_keep

df['review'] = df['review'].apply(clean_text)

def lemmatization(text):
    doc = nlp(text)

    # Converts reviews into lemmas (base-form words i.e. help reduce duplicate word forms)
    # Only keeping nouns and adjectives
    lemmas = [t.lemma_ for t in doc if t.pos_ in {"NOUN", "ADJ"}]
    return ' '.join(lemmas)

df['review'] = df['review'].apply(lemmatization)

df.to_csv('/content/DS-4002_Group5_Project1/DATA/McDonalds_Reviews_Cleaned.csv')
