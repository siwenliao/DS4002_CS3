# Goal: Fit the LDA model with the most optimal k value and assign a topic to each review
# Steps: Tokenize the reviews and create a document-term matrix to prepare for LDA, find the optimal k value based on coherence, 
# fit the LDA model with the chosen k value, assign a topic to each review, and analyze the topics for positive and negative reviews

# Uncomment the following line if you need to install to import gensim:
# !pip install --upgrade gensim pyLDAvis spacy pandas scikit-learn
import pandas as pd
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.models import CoherenceModel
import random, numpy as np
random.seed(100)
np.random.seed(100)

df = pd.read_csv('/content/DS-4002_Group5_Project1/DATA/McDonalds_Reviews_Cleaned.csv')
df1 = df[df['review'].notna()].copy()

# Turn 'review' column into a list of reviews
texts = df1['review'].tolist()

# For each individual review in the list, split the review (string) into a list of words
tokenized_text = [t.split() for t in texts]

# Assigns a number/ID to each word and a mapping for each review based on these IDs
dic = corpora.Dictionary(tokenized_text)

# Filtering out uncommon words to reduce runtime for next steps
dic.filter_extremes(no_below=10, no_above=0.5, keep_n=50000)

# Builds a bag-of-words matrix to input for LDA
doc_term_matrix = [dic.doc2bow(review) for review in tokenized_text]

# Finding the optimal k value to fit the LDA model
LDA = gensim.models.ldamodel.LdaModel

# Sampling the data to reduce runtime when fitting multiple LDA models with differen k values
# Once we choose a k value, we will fit the entire dataset 
N_sample = 5000
idx = random.sample(range(len(tokenized_text)), k=min(N_sample, len(tokenized_text)))
token_sample = [tokenized_text[i] for i in idx]
doc_sample = [doc_term_matrix[i] for i in idx]

# Fitting each LDA model with different k values
k_values = [3, 4, 5, 6, 7]
sample_models = {}
for k in k_values:
  sample_lda_model = LDA(
      corpus = doc_sample,
      id2word = dic,
      num_topics = k,
      random_state=100,
      chunksize=2000,
      passes=10,
      iterations = 100,
      alpha = 'auto',
      eta='auto'
  )
  sample_models[k] = sample_lda_model

# Finding the coherences of each LDA model with different k values
coherences = {}
for k, sample_model in sample_models.items():
  coherence_model_sample = CoherenceModel(
      model = sample_model,
      texts = token_sample,
      dictionary = dic,
      coherence = 'c_v'
  )
  coherences[k] = coherence_model_sample.get_coherence()

# Printing the coherence of each model for evaluation
print('Coherence scores: \n')
for k in sorted(coherences):
  print('k value:', k, 'Coherence:', coherences[k], '\n')

# LDA model with the entire dataset and chosen k value (k=6)

lda_model = LDA(
    corpus = doc_term_matrix,
    id2word = dic,
    num_topics = 6,
    random_state = 100,
    chunksize = 2000,
    passes = 10,
    iterations = 100
)
print(lda_model.print_topics(), '\n')

# Evaluating the model (perplexity and coherence)

print('\nPerplexity:', lda_model.log_perplexity(
        doc_term_matrix, total_docs=len(doc_term_matrix)))

coherence_model_lda = CoherenceModel(
        model=lda_model,
        texts=tokenized_text,
        dictionary=dic,
        coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence:', coherence_lda)

# Assigning the topic of highest probability to each review

def get_topic(model, review):
  topic_probs = model.get_document_topics(review)
  topic, max_prob = max(topic_probs, key=lambda x: x[1])
  return topic, max_prob

topic_all = [get_topic(lda_model, review) for review in doc_term_matrix]
# returns a list of (topic, probability)

# Adding columns in the dataset to record the topic and the corresponding probability
df1['topic'] = [t for t, p in topic_all]
df1['topic_probability'] = [p for t, p in topic_all]

# Giving the topic categories more descriptive labels

topic_labels = {
    0: 'Bad service, wrong orders, rude staff',
    1: 'Fast and quick service, clean place, friendly staff',
    2: 'Overall good price, quality, service',
    3: 'Food quality issues (cold/old fries, burgers, nuggets)',
    4: 'Long wait times, slow drive-thru',
    5: 'Miscellanous, general complains about environment'
}

df1['topic_label'] = df1['topic'].map(topic_labels)

# Extracting the number from the ratings (omitting the 'star' or 'stars' label that follows the number)
df1['rating_number_only'] = df1['rating'].str.extract(r'(\d)').astype(int)

# Assign positive, negative, and neutral sentiments to reviews by their rating
# 4 and 5 stars = positive, 1 and 2 stars = negative, 3 stars = neutral
def classify_rating(r):
  if r >= 4:
    return 'positive'
  elif r <= 2:
    return 'negative'
  else:
    return 'neutral'

# Adding a column to describe the sentiment of each review
df1['sentiment'] = df1['rating_number_only'].apply(classify_rating)

# Find common topics in positive and negative reviews

positive_topics = df1[df1['sentiment'] == 'positive']['topic_label'].value_counts()
negative_topics = df1[df1['sentiment'] == 'negative']['topic_label'].value_counts()
neutral_topics = df1[df1['sentiment'] == 'neutral']['topic_label'].value_counts()

print('Most common topics in positive reviews: \n', positive_topics, '\n')
print('Most common topics in negative reviews: \n', negative_topics, '\n')
print('Most common topics in neutral reviews: \n', neutral_topics, '\n')

# Proportion of each topic in positive and negative reviews

positive_topic_proportion = df1[df1['sentiment'] == 'positive']['topic_label'].value_counts(normalize=True)
negative_topic_proportion = df1[df1['sentiment'] == 'negative']['topic_label'].value_counts(normalize=True)

print('Topic distribution for positive reviews: \n', positive_topic_proportion, '\n')
print('Topic distribution for negative reviews: \n', negative_topic_proportion, '\n')

# Saving dataset
df1.to_csv('/content/DS-4002_Group5_Project1/DATA/McDonalds_Reviews_With_Topics.csv', index=False)
