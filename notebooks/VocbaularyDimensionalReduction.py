import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
import gensim

import sys

sys.path.append('notebooks/Functions')
from TextMiningProcesses import column_lemmatizer

# import re

# #Headers have already been tokenized and had stopwords removed, but they have not been lemmatized
# # headers need to be lemetized, and made unique again

filepaths = [
    'data/vocab_df_part_0.csv',
    'data/vocab_df_part_1.csv',
    'data/vocab_df_part_2.csv',
    'data/vocab_df_part_3.csv',
    'data/vocab_df_part_4.csv',
    'data/vocab_df_part_5.csv',
    'data/vocab_df_part_6.csv',
    'data/vocab_df_part_7.csv',
    'data/vocab_df_part_8.csv',
    'data/vocab_df_part_9.csv',
    'data/vocab_df_part_10.csv',
    'data/vocab_df_part_11.csv'
]

vocab_list = []

for filepath in filepaths:
    headers = pd.read_csv(filepath, nrows=0).columns.tolist()

    vocab_list.extend(headers)

vocab_list = list(set(vocab_list))

# print(len(vocab_list))

lemmatizer = WordNetLemmatizer()
vectorizer = CountVectorizer(stop_words='english')

lemmed = ''

for word in vocab_list:

    lemmed_word = lemmatizer.lemmatize(word)
    lemmed += lemmed_word + ' '

lemmed.strip()

# print(len(lemmed))

vocab_df = pd.DataFrame()
vocab_df.columns = lemmed

vocab_df.to_csv('data/vocab.csv')

print(len(vocab_df.columns.head()))

df = pd.read_json("../../data/Appliances.json", lines = True)
df = df.dropna(subset='reviewText')

chunk_size = len(df) / 20

with tqdm(total=20) as pbar:
    for chunk_number in range(20):
        start_index = chunk_number * chunk_size
        end_index = (chunk_number + 1) * chunk_size
        chunk = df.iloc[start_index:end_index]

        features = chunk['reviewText']
        target = chunk['overall']

        lemmed_features = column_lemmatizer(features)

        vectored_features = vectorizer.fit_transform(lemmed_features)

        # Make a dataframe for machine learning
        chunk_df = pd.DataFrame(chunk.toarray(), columns=vectorizer.get_feature_names_out())

        if chunk_df.columns not in vocab_df.columns:
            raise ValueError('Chunk columns in {start_index} do not match vocab_df')
            break

    pbar.update(1)
