def text_stemmer(tokens, stemmer, go_gauge):
    from nltk.stem import WordNetLemmatizer
    from nltk.stem.snowball import EnglishStemmer
    if stemmer == 'lemmatizer':
        lemmatizer = WordNetLemmatizer()
        stem_string = ''
        for token in tokens:
            if token and token in go_gauge:
                stemmed_word = lemmatizer.lemmatize(token)
                stem_string += stemmed_word + ' '    
        stem_string = stem_string.strip()
        return stem_string
    elif stemmer == 'stemmatizer':
        stemmat = EnglishStemmer()
        stem_string = ''
        for token in tokens:
            if token and token in go_gauge:
                stemmed_word = stemmat.stem(token)
                stem_string += stemmed_word + ' '    
        stem_string = stem_string.strip()
        return stem_string
    else:
        raise ValueError('Unsupported text stemmer passed')
        
def preprocess_items(text, tokenizer, spell_checker):
    import re
    import pandas as pd
    import nltk
    from nltk.tokenize.regexp import RegexpTokenizer

    if isinstance(text, str):
        item_no_tags = re.sub(r'<.*?>.*?<.*?>', '', text)
        item_tokens = tokenizer.tokenize(item_no_tags.lower())
        token_list = []
        for token in item_tokens:
            corrected_token = spell_checker.correction(token.strip())
            token_list.append(corrected_token)
        return token_list
    
def process_item(text, tokenizer, spell_checker, stemmer, go_gauge):
    tokens = preprocess_items(text=text, tokenizer=tokenizer, spell_checker=spell_checker)
    stemmed_string = text_stemmer(tokens=tokens, stemmer=stemmer, go_gauge=go_gauge)
    if stemmed_string is None:
        stemmed_string = ''
    return stemmed_string

def process_item_wrapper(args):
    item = args[0]
    tokenizer = args[1]
    spell_checker = args[2]
    stemmer = args[3]
    go_gauge = args[4]
    return process_item(text=item, tokenizer=tokenizer, spell_checker=spell_checker, stemmer=stemmer, go_gauge=go_gauge)
            
def new_column_lemmatizer(text_series):
    """
    This function preprocesses a pandas Series of sentences, typically taken from a dataframe column and prepares them for classification/regression modelling
    by tokenizing, removing stop words, and lemmatizing the series.
    
    Args:
    text_series (pd.Series): Input pandas Series containing sentences.
    
    Returns:
    pd.Series: Processed Series containing lemmatized words.

    Example:
    df['to_be_lemmed'] = pd.Series({0: 'I like this', 1: 'good times'})
    df['lems'] = column_stemmatizer(df['to_be_lemmed'])

    returns:
    df['lems'] = pd.Series({0: ['like'], 1: ['good', 'time']})
    """
    import pandas as pd
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize.regexp import RegexpTokenizer
    from spellchecker import SpellChecker
    from tqdm import tqdm
    import csv

    from spellchecker import SpellChecker
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor
    
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    spell_checker = SpellChecker()
    # Download NLTK resources
    if not nltk.corpus.stopwords.fileids():
        nltk.download('punkt')
        nltk.download('stopwords')
  
    # Initialize the lemmatizer and stopwords set
    stop_words = set(stopwords.words('english'))
    with open('best_features_list.csv', 'r') as file:
        reader = csv.reader(file)
        go_gauge = next(reader)

    if isinstance(text_series, pd.Series):
        args_list = [
            (item, tokenizer, spell_checker, 'lemmatizer', go_gauge)
            for item in text_series
        ]

        # If 'reviewText' is a list, apply the function to each element of the list
        with ProcessPoolExecutor() as executor:
            lemmed_cells = list(executor.map(process_item_wrapper, args_list)
            )

            lemmed_series = pd.Series(lemmed_cells)

            return lemmed_series
    else:
        raise TypeError('function must take a pd.Series as argument')

def new_column_stemmatizer(text_series):
    """
    This function preprocesses a pandas Series of sentences, typically taken from a dataframe column and prepares them for classification/regression modelling
    by tokenizing, removing stop words, and stemmatizing the series.
    
    Args:
    text_series (pd.Series): Input pandas Series containing sentences.
    
    Returns:
    pd.Series: Processed Series containing stemmed words.

    Example:
    df['to_be_stemmed'] = pd.Series({0: 'I like this', 1: 'good times'})
    df['stems'] = column_stemmatizer(df['to_be_stemmed'])

    returns:
    df['stems'] = pd.Series({0: ['like'], 1: ['good', 'time']})
    """
    import pandas as pd
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize.regexp import RegexpTokenizer
    from spellchecker import SpellChecker
    from tqdm import tqdm
    import csv

    from spellchecker import SpellChecker
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor
    
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    spell_checker = SpellChecker()
    # Download NLTK resources
    if not nltk.corpus.stopwords.fileids():
        nltk.download('punkt')
        nltk.download('stopwords')
  
    # Initialize the lemmatizer and stopwords set
    stop_words = set(stopwords.words('english'))
    with open('best_features_list.csv', 'r') as file:
        reader = csv.reader(file)
        go_gauge = next(reader)

    if isinstance(text_series, pd.Series):
        args_list = [
            (item, tokenizer, spell_checker, 'stemmatizer', go_gauge)
            for item in text_series
        ]

        # If 'reviewText' is a list, apply the function to each element of the list
        with ProcessPoolExecutor() as executor:
            lemmed_cells = list(executor.map(
                process_item_wrapper, args_list
                )
            )

            lemmed_series = pd.Series(lemmed_cells)

            return lemmed_series
    else:
        raise TypeError('function must take a pd.Series as argument')

# # Test Elements



# Fit and transform the specified text data

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_extraction.text import CountVectorizer

    # Create a TfidfVectorizer instance
    vectorizer = CountVectorizer()

    X = pd.Series({
        0: 'good, I like it', 
        1: '<http. vanilla> happy best*&^#*(&t',
        2: 'ceramic certain juice rtequila',
        3: '',
        4: '9999',
        5: 'good product'
    })

    lem_X = new_column_stemmatizer(X)

    print(lem_X)

    vectors = vectorizer(lem_X)

    vector_df = pd.DataFrame(vectors.todense(), columns=vectorizer.get_feature_names_out())

    print(vector_df[0].shape)

    lr = LinearRegression()

    y = pd.Series({
        0: 1, 
        1: 1,
        2: 1,
        3: 0,
        4: 0,
        5: 0
    })

    lr.fit(vectors[0], y)

    print(lr.score(vectors[0], y))

