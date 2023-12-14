import sys
sys.path.append('../notebooks/Functions')

from NewTextMiningProcesses import new_column_lemmatizer, new_column_stemmatizer, new_count_vectorize_data, new_tfidf_vectorize_data


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import pickle
import joblib
import os

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('../data/train.csv')
    return df
@st.cache_data
def load_best_models():
    with open('../notebooks/Classification/RFBestModel.pkl', 'rb') as file:
        rf_model = pickle.load(file)
    with open('../notebooks/HGBRBestModel.pkl', 'rb') as file:
        hgbr_model = pickle.load(file)
    with open('../notebooks/Classification/LogisticBestModel.pkl', 'rb') as file:
        logistic_model = pickle.load(file)
    return rf_model, hgbr_model, logistic_model

st.set_option('deprecation.showPyplotGlobalUse', False)

df = load_data()
rf_model, hgbr_model, logistic_model = load_best_models()

# Set the title and sidebar
st.title("Comparison of Text Mined Customer Review Rating Prediction Models ")
st.sidebar.title("Table of Contents")
pages = ["Project Goals", "DataSet Quality", "Machine Learning Methodologies", "Application", "Conclusions and Next Steps"]
page = st.sidebar.radio("Go to", pages)

if page == pages[0]:
    st.title('Project Goals')
    st.markdown('## Abstract')
    st.markdown('This report compares methods of datamining the text from Customer Reviews and \
            the Accuracy of Rating predictions from 1 to 5 stars. This report satisfies the \
            business need to identify the best model to accurately classify Customer reviews \
            into a rating. The report aims to save future studies time in computation \
            comparisons by finding the best preprocessing methods and the best models to \
            classify reports by rating.  ')
    st.markdown('The findings of this report could be implemented in several use cases:  ')
    st.markdown("* Generate an automated rating system, which offers customers a pre-generated \
            star rating based on the content of their review.   ")
    st.markdown("* Identify Reviews which are incorrectly rated, to remove them from further \
            analyses or submit them to further analyses.   ")
    st.markdown("""* Classify reviews which are no longer associated with their original rating,
    or reviews which are not part of a rating system.  
    * Sort reviews for customer service customer response Management to organize reviews by priority.  
    * Classify reviews for automated CRM Tools enabling automated responses to reviews based on predicted rating.""")
    st.markdown('## Introduction')
    st.markdown("This report analyses data from Amazon reviews from the Appliances Category, \
             the data was originally collected in 2014 and most recently updated in 2018. \
             Though the data has been parsed for NLP usage, extra Data Cleaning and preprocessing \
             is required to enable the variety of modelling techniques that will be tested in \
             the main report.  ")
    st.markdown("The features will be derived from the review text of each review in the data and the \
             target variable is the rating from 1-5 stars. In this report, the feature text data \
             will be checked for duplicates and NaN values, and vectorized.  ")
    st.markdown("Other columns will also be processed to enable further investigations and project \
             expansions.")

if page == pages[1] :
    @st.cache_data
    def load_synopsis():
        column_dict = {
        "Column Heading": ['overall', 'vote', 'verified', 'reviewTime', 'reviewerID', 'asin', 'style',
                           'reviewerName', 'reviewText', 'summary', 'unixreviewTime', 'image'],
        "Brief": ['TARGET – star ratings valued 1-5 ', 'Number of upvotes granted to the review ',
                  'Identifies verified buyers', 'Datetime review was left', 'Customer UID',
                  'Item reference number', 'String holding various item specifics', 'Review info',
                  'FEATURE - Text based review', 'Potential secondary Feature – review summary',
                  'Datetime for review in unix time', 'Review images if included in review'],
        "Number of Null Records": [0, 537515, 0, 0, 0, 0, 464804, 15, 324, 128, 0, 593519]
        }

        column_df = pd.DataFrame(column_dict)
        column_df = column_df.set_index("Column Heading")
        return column_df
    
    @st.cache_data
    def ratings():
        rating_dict = {
            "overall (Rating column - Target)": [5, 4, 3, 2, 1],
            "Percentage of Reviews": ['69%', '13%', '5%', '3%', '10%']
        }
        rating_df = pd.DataFrame(rating_dict)
        rating_df = rating_df.set_index("overall (Rating column - Target)")
        return rating_df

    st.title("Dataset Quality")
    st.markdown('## Data Source')
    st.markdown("#### Source Citation:")
    st.markdown("""> **Justifying recommendations using distantly-labeled reviews and fined-grained aspects**  
> Jianmo Ni, Jiacheng Li, Julian McAuley  
> _Empirical Methods in Natural Language Processing (EMNLP), 2019_""")
    st.markdown('From this data, only the appliance reviews were utilized for this project. These were \
             selected as the number of records is between 500k and 1m records. This criterion was \
             estimated to ensure that enough data is present to produce a well-fitting model, but \
             not so many that computational times will exceed 24 hours.')
    st.markdown('## Data Overview')
    st.dataframe(load_synopsis())
    st.markdown('### Target Data Balance')
    st.dataframe(ratings())
    st.markdown('The target column ‘overall’ is imbalanced, as shown by the graph above. This indicates \
             that the data will need to be sampled to increase classification accuracy. 69% of the \
             reviews awarded a rating of 5, which indicates that a very basic model that simply calls \
             every review 5 stars will be 69% accurate against this data.')
    st.markdown(f"""> ##### Our baseline Accuracy is 69%
> assuming all reviews are assigned rating 5.""")
    st.image('../images/ReportImages/OverallRatingDistribution.jpg', caption='Rating Distribution', use_column_width=True)
    st.markdown('### Dataset Preprocessing')
    st.markdown("To achieve the best possible results from the dataset it is essential to reduce and format\
            the dataset into a data type and format that enables the models to genersate the best possible\
            accuracies.  ")
    st.markdown("The reviews include html tags for videos and images if any were included in the review \
            This text must be removed, as must any text including numbers, misspellings must be converted \
            to the correct spelling and tokenised. To achieve these results, Regex, pyspellchecker and the \
            RegExTokenizer were used to pre process the text in each review In addition to this the reviews\
            were compared to nltk's English stopwords and stopwords were removed.")
    st.markdown('Several further methods were implemented to ready the dataset for modelling. These included \
            WordNetLemmatizer and the ENglishStemmer from the nltk.stem library. These models reduce the words to \
            roots of the word in different formats. To enable machine learning on these stemming methods, \
            the datasets need to be converted to number vectors and a further two models in the CountVectorizer \
            and TFIDF Vectorizer from sci-kit Learns text library.')
    st.markdown("The data was also vectorized using Google's Word2Vec model, which did not use the stemmers to produce \
            vectors.")
    st.markdown("## A Note on Sampling")
    st.markdown("It is obvious that the dataset is imbalanced, analysis was also performed on 4 samplers to see \
            which would be the most beneficial to the models, however the model quality was too poor for the \
            sampling results to be valid and so sampling was not well rated. Instead models are attuned on raw \
            unless an improvement is seen through smapling and oversampling techniques are largely dropped due \
            to data/working memory limitations.")
    st.markdown('Although the samplers proved to be unuseful, each one had a gridsearch performed to identify the maximum \
            potential sample with the given parameters.')
    
    options = {
        'RandomOverSampler, RandomUndersampler, no sampling',
        'Synthetic Minority Over Sampling Technique',
        'Cluster Centroids'
    }

    sampler_display = st.radio('The tested parameters included:', (
        'RandomOverSampler, RandomUndersampler, no sampling',
        'Synthetic Minority Over Sampling Technique',
        'Cluster Centroids'
    ))
    if sampler_display == 'RandomOverSampler, RandomUndersampler, no sampling':
        st.markdown('For the simpler models a simple best accuracy with the selected Sampler was taken. \
                For models with several arguments to be tested, these were tested simultaneously with \
                the sampler gridsearch to save computation in later more complex tests. These parameters \
                were graphed when available.')
        sample_display = st.radio('Which model do you want to view?', (
            'Linear Regression',
            'Lasso Regression',
            'Ridge Regression',
            'ElasticNet Regression',
            'HGBC Regression'))
        if sample_display == 'Linear Regression':
            st.markdown("""The best results were returned by the English Stemmer TFIDF Vector text processes which were sampled by the RandomOverSampler.  
* Accuracy: 0.525
* Mean Squared Error: 2.020""")
        elif sample_display == 'Lasso Regression':
            st.markdown("""The best Lasso results were provided by the lemmatized Count Vector processes with no sampling.  
* Accuracy: 0.284
* Mean Squared Error: 0.271""")
            st.markdown('Lasso alpha values between 0.001 and 0.3 were tested simultaneously.')
            st.image('../images/lasso_alphas.png', use_column_width=True)
        elif sample_display == 'Ridge Regression':
            st.markdown("""The best Ridge results were provided by:  
* Lemmatizer Count Vector with RandomOverSampler sampling:  
    * Accuracy: 0.557  
    * Mean Squared Error: 2.915  
* Lemmatizer TFIDF Vector with RandomOverSampler sampling:
    * Accuracy: 0.478
    * Mean Squared Error: 1.11""")
            st.markdown('Ridge alpha values between 0.001 and 0.3 were tested simultaneously. Though \
                    0.001 has the best accuracy, 0.3 was take forward as the Mean Squared Error nearly tripled \
                    at lower alphas.')
            st.image('../images/ridge_alphas.png', use_column_width=True)
        elif sample_display == 'ElasticNet Regression':
            st.markdown("""The best ElasticNet results were provided by:  
* Lemmatizer Count Vector with RandomOverSampler sampling:  
    * Accuracy: 0.318  
    * Mean Squared Error: 1.406  
* Lemmatizer Count Vector with no sampling:  
    * Accuracy: 0.296  
    * Mean Squared Error: 1.577""")
            st.markdown('ElasticNet alpha values between 0.001 and 0.3 and L1 ratio values between 0.3 and 0.7 were \
                    tested simultaneously.')
            st.image('../images/enet_alphas.png', use_column_width=True)
            st.image('../images/enet_l1_ratios.png', use_column_width=True)
        elif sample_display == 'HGBC Regression':
            st.markdown("""The best HGBC results were provided by:  
* Lemmatizer TFIDF Vector with RandomOverSampler sampling:  
    * Accuracy: 0.499  
    * Mean Squared Error: 1.351  
* Lemmatizer TFIDF Vector with no sampling:  
    * Accuracy: 0.410  
    * Mean Squared Error: 1.351""")
            st.markdown('HGBC learning rate values between 0.1 and 0.5 and max depths between 50 and 1000 were \
                    tested simultaneously.')
            st.image('../images/hgbr_learn_rate.png', use_column_width=True)
            st.image('../images/hgbr_max_depth.png', use_column_width=True)
    elif sampler_display == 'Synthetic Minority Over Sampling Technique':
        sample_display = st.radio('Which model do you want to view?', (
            'Linear Regression',
            'Lasso Regression',
            'Ridge Regression',
            'ElasticNet Regression',
            'HGBC Regression'))
        if sample_display == 'Linear Regression':
            st.markdown("""Smote K_Neighbours were tested between 5 and 1000:  
* Lemmatizer TFIDF Vector - 500 k_neighbors  
    * Accuracy: 0.566  
    * Mean Squared Error: 2.96  
* Lemmatizer TFIDF Vector - 1000 k_neighbors  
    * Accuracy: 0.535  
    * Mean Squared Error: 1.98""")
            st.markdown('Though 500 k_neighbors has the best accuracy, 1000 k_neighbors were taken forward as the \
                     Mean Squared Error and R Squared values (R Squared not shown) performed marginally better.')
            st.image('../images/lr_smote_k_neighbors.png', use_column_width=True)
        elif sample_display == 'Lasso Regression':
            st.markdown("""Smote K_Neighbours were tested between 5 and 1000:  
* English Stemmer TFIDF Vector - 500 k_neighbors  
    * Accuracy: 0.224  
    * Mean Squared Error: 1.61  
* Lemmatizer Count Vector - 1000 k_neighbors  
    * Accuracy: 0.211  
    * Mean Squared Error: 1.54""")
            st.image('../images/lasso_smote_k_neighbors.png', use_column_width=True)
        elif sample_display == 'Ridge Regression':
            st.markdown("""Smote K_Neighbours were tested between 5 and 1000:  
* Lemmatizer TFIDF Vector - 1000 k_neighbors  
    * Accuracy: 0.490  
    * Mean Squared Error: 1.081""")
            st.image('../images/ridge_smote_k_neighbors.png', use_column_width=True)
        elif sample_display == 'ElasticNet Regression':
            st.markdown("""Smote K_Neighbours were tested between 5 and 1000:  
* Lemmatizer Count Vector - 1000 k_neighbors  
    * Accuracy: 0.237  
    * Mean Squared Error: 1.489""")
            st.image('../images/enet_smote_k_neighbors.png', use_column_width=True)
        elif sample_display == 'HGBC Regression':
            st.markdown("""Smote K_Neighbours were tested between 5 and 1000:  
* Lemmatizer Count Vector - 50 k_neighbors  
    * Accuracy: 0.486  
    * Mean Squared Error: 1.188  
* Lemmatizer TFIDF Vector - 250 k_neighbors  
    * Accuracy: 0.443  
    * Mean Squared Error: 1.184  
* Lemmatizer TFIDF Vector - 100 k_neighbors  
    * Accuracy: 0.0.446  
    * Mean Squared Error: 1.176""")
            st.image('../images/hgbr_smote_k_neighbors.png', use_column_width=True)
    elif sampler_display == 'Cluster Centroids':
        st.markdown('Although it was suspected that the Cluster Centroids sampler would perform well\n \
                if the dataset responds well to classification techniques, the Cluster Centroids method had no \
                effect on most of the models, only the HGBR model showed any change, and even that change was insignificant.')
        sample_display = st.radio('Which model do you want to view?', (
            'Linear Regression',
            'Lasso Regression',
            'Ridge Regression',
            'ElasticNet Regression',
            'HGBC Regression'))
        if sample_display == 'Linear Regression':
            st.image('../images/lr_cc_n_clusters.png', use_column_width=True)
        elif sample_display == 'Lasso Regression':
            st.image('../images/lasso_cc_n_clusters.png', use_column_width=True)
        elif sample_display == 'Ridge Regression':
            st.image('../images/ridge_cc_n_clusters.png', use_column_width=True)
        elif sample_display == 'ElasticNet Regression':
            st.image('../images/enet_cc_n_clusters.png', use_column_width=True)
        elif sample_display == 'HGBC Regression':
            st.image('../images/hgbr_cc_n_clusters.png', use_column_width=True)
    
if page == pages[2]:
    @st.cache_data
    def Regression_Bests():
        regression_dict = {
            "Model": ['HGBR', 
                      'Ridge',
                      'Linear Regression',
                      'Lasso:',
                      'ElasticNet:'
            ],
            "Token Method": [
                'lemmatized',
                'lemmatized',
                'English Stemmer',
                'lemmatized',
                'lemmatized',
            ],
            "Vector Method": [
                'TFIDF Vector',
                'TFIDF Vector',
                'TFIDF Vector',
                'Count Vector',
                'Count Vector'
            ],
            "Sampler": [
                'None',
                'Smote',
                'Smote',
                'None',
                'RandomOverSampler',
            ],
            "Mean Train Accuracy": [0.604, 0.523, 0.697, 0.294, 0.303],
            "Mean Test Accuracy": [0.593, 0.382, 0.380, 0.280, 0.269],
            "Mean Train Precision": [0.740, 0.750, 0.802, 0.659, 0.680],
            "Mean Test Precision": [0.728, 0.643, 0.512, 0.628, 0.615],
            "Mean Train Recall": [0.604, 0.523, 0.697, 0.294, 0.303],
            "Mean Test Recall": [0.593, 0.382, 0.380, 0.280, 0.269],
            "Mean Train F1 Score": [0.642,0.562, 0.724, 0.322, 0.331],
            "Mean Test F1 Score": [0.632, 0.435, 0.423, 0.303, 0.300],
            "Mean Train R Squared": [0.563, 0.892, 0.838, 0.488, 0.507],
            "Mean Test R Squared": [0.509, 0.510, 0.162, 0.444, 0.423],
            "Mean Train Mean Squared Error": [0.747, 0.700, 0.465, 1.470, 1.415],
            "Mean Test Mean Squared Error": [0.829, 1.393, 2.384, 1.580, 1.641],
        }
        regression_df = pd.DataFrame(regression_dict)
        return regression_df
    
    @st.cache_data
    def Classification_Bests():
        classification_dict = {
            "Model": ['SVM', 
                      'Logistic Regression',
                      'Random Forest',
                      'HGBC',
                      'Decision Tree',
                      'Naive bayes',
                      'K Nearest Neighbor',
            ],
            "Token Method": [
                'lemmatized',
                'lemmatized',
                'lemmatized',
                'English Stemmer',
                'lemmatized',
                'English Stemmer',
                'English Stemmer',
            ],
            "Vector Method": [
                'TFIDF Vector',
                'TFIDF Vector',
                'Count Vector',
                'Count Vector',
                'Count Vector',
                'Count Vector',
                'Count Vector',
            ],
            "Sampler": [
                'None',
                'None',
                'None',
                'None',
                'None',
                'RandomUnderSampler',
                'RandomUnderSampler'
            ],
            "Mean Train Accuracy": [0.874, 0.769, 0.971, 0.782, 0.971, 0.523, 0.556],
            "Mean Test Accuracy": [ 0.770, 0.762, 0.753, 0.742, 0.683, 0.624, 0.484],
            "Mean Train Precision": [0.882, 0.720, 0.971, 0.752, 0.971, 0.517, 0.556],
            "Mean Test Precision": [0.725, 0.703, 0.708, 0.682, 0.663, 0.711, 0.660],
            "Mean Train Recall": [0.814, 0.769, 0.971, 0.782, 0.971, 0.523, 0.556],
            "Mean Test Recall": [0.770, 0.763, 0.753, 0.742, 0.683, 0.624, 0.660],
            "Mean Train F1 Score": [0.856, 0.715, 0.970, 0.745, 0.970, 0.514, 0.552],
            "Mean Test F1 Score": [0.709, 0.707, 0.684, 0.698, 0.673, 0.659, 0.541],
            "Mean Train R Squared": [0.762, 0.408, 0.955, 0.407, 0.954, 0.311, 0.232],
            "Mean Test R Squared": [0.410, 0.376, 0.267, 0.299, 0.158, 0.111, -0.214],
            "Mean Train Mean Squared Error": [0.403, 1.004, 0.076, 1.00, 0.076, 1.378, 1.537],
            "Mean Test Mean Squared Error": [0.996, 1.053, 1.24, 1.18, 1.421, 1.501, 2.050],
        }
        classification_df = pd.DataFrame(classification_dict)
        return classification_df
    
    @st.cache_data
    def concat_two_dfs_vertical():
        df1 = Regression_Bests()
        df2 = Classification_Bests()
        concat_df = pd.concat([df1, df2])
        train_df = concat_df[[
            'Model',
            'Token Method', 
            'Vector Method', 
            'Sampler', 
            "Mean Train Accuracy",
            "Mean Train Precision",
            "Mean Train Recall",
            "Mean Train F1 Score",
            "Mean Train R Squared",
            "Mean Train Mean Squared Error"]].copy()
        train_df['Accuracy Type'] = 'Train'
        train_df = train_df.rename(columns={"Mean Train Accuracy": "Accuracy",
            "Mean Train Precision": "Precision",
            "Mean Train Recall": "Recall",
            "Mean Train F1 Score": "F1 Score",
            "Mean Train R Squared": "R Squared",
            "Mean Train Mean Squared Error": "Mean Squared Error"})
        test_df = concat_df[[
            'Model',
            'Token Method', 
            'Vector Method', 
            'Sampler', 
            "Mean Test Accuracy",
            "Mean Test Precision",
            "Mean Test Recall",
            "Mean Test F1 Score",
            "Mean Test R Squared",
            "Mean Test Mean Squared Error"]].copy()
        test_df['Accuracy Type'] = 'Test'
        test_df = test_df.rename(columns={"Mean Test Accuracy": "Accuracy",
            "Mean Test Precision": "Precision",
            "Mean Test Recall": "Recall",
            "Mean Test F1 Score": "F1 Score",
            "Mean Test R Squared": "R Squared",
            "Mean Test Mean Squared Error": "Mean Squared Error"})
        concat_with_acc_type = pd.merge(train_df, test_df, how='outer')
        grouped_concat = concat_with_acc_type.groupby(['Model', 'Token Method', 'Vector Method', 'Sampler', 'Accuracy Type']).max()
        grouped_concat = grouped_concat[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'R Squared', 'Mean Squared Error']]
        grouped_concat = grouped_concat.sort_values(by="Accuracy", ascending=False).reset_index()
        return grouped_concat
    
    @st.cache_data
    def Word2Vec_df():
        dfw2v_dict = {
            "Model": ['Logistic Regression',
                      'Random Forest'
            ],
            "Vector Method": [
                'Google Word 2 Vector',
                'Google Word 2 Vector'
            ],
            "Sampler": [
                'RandomOverSampler',
                'RandomOverSampler'
            ],
            "Mean Test Accuracy": [0.526, 0.651],
            "Mean Test Precision": [0.60, 0.55],
            "Mean Test Recall": [0.53, 0.65],
            "Mean Test Specificity": [0.70, 0.36],
            "Mean Test F1 Score": [0.56, 0.57],
            "Mean Test Geometric Mean Error": [0.57, 0.26],
            "Mean Test Mean Squared Error": ['-', 2.31]
        }
        dfw2v = pd.DataFrame(dfw2v_dict)
        return dfw2v
    
    @st.cache_resource()
    def plot_and_show_accuracy_by_model():
        data = concat_two_dfs_vertical()
        fig = sns.catplot(x='Model', y='Accuracy', hue='Accuracy Type', data=data, kind='bar')
        plt.xticks(rotation=90)
        plt.title('Accuracies by Model')

        return fig
    
    st.title("Modelling")
    st.markdown('In this report several Classification and Regression Models are compared using data \
            from amazon reviews as detailed in the Data Quality Report.')
    st.markdown('## Classification Models')
    st.markdown("""The Models being compared include:
* LogisticRegression
* Support Vector Machine Classification
* K Nearest Neighbors
* Decision Tree Classifier
* Random Forest Classifier
* Naïve Bayes
* Histogram Gradient Boosting Classifier""")
    st.markdown('## Regression Models')
    st.markdown("""The Models being compared include:
* Linear Regression
* Lasso
* Ridge
* ElasticNet 
* Histogram Gradient Boosting Regressor""")
    st.markdown('## Hypothesis')
    st.markdown('The ratings target variable consists of integers that follow a linear related scale. \
            This means that they should be able to return a reliable score as well as a classification \
            model. This project aims to compare the accuracy of this statement using a comprehensive number of \
            preprocessing and modelling techniques.')
    st.markdown('Whilst the classification models return exact categories 1-5, the regression models will only \
            return a continuous series of results. This has its benefits; the spread of the data can be \
            analyzed in greater detail than the classification reports. The visibility of the data can be used \
            to identify edge cases and see how noise in the categories behaves, which is not possible in the \
            classification reports.')
    st.markdown('The HistGradientBoostingRegressor (HGBR) and the HistGradientBoostingClassifier (HGBC) were \
            chosen over the GradientBoostingRegressor (GBR) and the GradientBoostingClassifier (GBC) to reduce \
            runtimes whilst operating on reasonably sized datasets.')
    st.markdown('## Model Comparisons')
    st.dataframe(concat_two_dfs_vertical())
    st.pyplot(plot_and_show_accuracy_by_model())
    st.markdown('## Results Analysis')
    st.markdown("The results clearly show that Classification techniques are more suited to the text data, however \
            none of the results are strong enough to clearly define the models as better than a basic model. The \
            best model produces only 77% accurate results with a large mean squared error of 0.996 on the test data. \
            This means that the average std deviation covers 1 class to either side of the corect class and the \
            confusion Matrices show that this accuracy largely relies on putting a large percentage of the data in \
            class 5.")
    st.markdown("77% Accuracy is only 8% better than a basic model that returns class 5 for every prediction and so this model \
            can not be considered very strong.")
    st.image('../images/SVMBestConfMatrix.png', caption='Best Model Accuracy (SVM) Confusion Matrix', use_column_width=True)
    st.markdown('### Google Word 2 Vector Analysis')
    st.markdown("The google Word 2 Vector Vectorizer is a text processor that assesses word relationships to support the \
            google search engine. Though the model is designed for search engine modelling rather than sentiment \
            analysis, the model was used to preprocess the review text to identify if the connections between words \
            in the model could provide a better model accuracy.")
    st.markdown("The word 2 vec preprocessor works on complete sentences rather than tokens like the previous preprocessing \
            techniques used. Each word in the sentence is converted into a vector of length 300. The vectors for each \
            word are added together to make one vector with 300 features representing the sentence.")
    st.markdown("The main advantage of this text process is that 300 features are processed much much faster than \
            a sparse matrix or dense matrix containing between 10,0000 and 70,000 features, depending on the contributing \
            processes and the number of records used to train the dataset.")
    st.markdown("Due to time constraints in the project, only two models with low processing times were modelled with the \
            word2vec text process, and no word stemming was used.")
    st.dataframe(Word2Vec_df())
    st.image('../images/lrW2VConfMatrix.png', caption='Logistic Regression Word 2 Vector Confusion Matrix', use_column_width=True)
    st.image('../images/rfW2VConfMatrix.png', caption='Random Forest Word 2 Vector Confusion Matrix', use_column_width=True)
    st.markdown('Like the regression modelling, the score has worsened in comparison to the target baseline accuracy of 69%. \
            This is most likely due to the mismatch of the search engine preprocessing model and the sentiment analysis \
            modelling that is being performed. It is possible to customize the Word 2 Vec process and this is likely a \
            good starting point for further analysis.')
    st.markdown('## Continuing the Investigation')
    st.markdown('As even the best model is not particularly strong more work must be performed to discover a \
            suitable model strength and provide accurate review ratings, or accurate sentiment analysis. \
            The next logical steps include improving the preprocessing methods or improving the machine \
            learning models that generated these results.')
    st.markdown('### Improving Preprocessing')
    st.markdown("Sampling had a negative affect on the accuracy results and as such sampling was omitted from the finalized accuracies. \
            This can only be because important features in the dataset were becoming diffused either by overproduction \
            of unimportant features, or through removal of important features.")
    st.markdown("This indicates that feature selection could yield better accuracies, by removing the diffusion and \
            allowing the modelling processes to better analysze the data.")
    st.markdown('### Improving Modelling')
    st.markdown('Another tool which has not yet been applied to the dataset is the SHAP analysis tool. \n \
            is \
            Alternatively, more advanced modelling could be implemented to find deeper relations between the words as \
            features. More complex models such as Deep Learning models could be implemented for such a purpose.')

if page == pages[3]:
    rf_model, hgbr_model, logistic_model = load_best_models()
    st.title("Applications")
    st.markdown('Understanding the sentiment of customers is an incredibly important part of customer relation management. \
            Even a basic model could significantly help a relations manager to identify and prioritize which customers to \
            respond to. When combined with a tool that identifies key items related to overall marketing plans such a tool \
            could greatly affect the success of the marketing strategy.')
    st.markdown('Additional use cases could include a tool which predicts the rating of a review either for the customer or more \
            for a company who have unrated reviews, either through lack of rating system or through data quality issues.')
    st.markdown('# Example Model')
    model_type_selection = st.selectbox("Choose a model type:", ["Classification", "Regression"])
    if model_type_selection == "Classification":
        model_selection = st.selectbox('Use the model below to compare modelling results:', ["Logistic Regression", "Random Forest"])
        if model_selection == "Logistic Regression":
            user_review = st.text_area("Enter your review here:", "I think this product is great! Well built and sturdy!")
            if st.button("Generate Prediction"):
                lem_review = new_column_lemmatizer(user_review)
                cv_review = new_tfidf_vectorize_data(lem_review)
                [user_result] = logistic_model.predict(cv_review)
                [[class_1_confidence, class_2_confidence, class_3_confidence, class_4_confidence, class_5_confidence]] = logistic_model.predict_proba(cv_review)
                st.markdown('### Model Prediction')
                st.markdown(f'#### Class: {user_result}')
                if user_result == 1:
                    st.markdown(f'#### Confidence: {round(class_1_confidence * 100, 1)}%')
                if user_result == 2:
                    st.markdown(f'#### Confidence: {round(class_2_confidence * 100, 1)}%')
                if user_result == 3:
                    st.markdown(f'#### Confidence: {round(class_3_confidence * 100, 1)}%')
                if user_result == 4:
                    st.markdown(f'#### Confidence: {round(class_4_confidence * 100, 1)}%')
                if user_result == 5:
                    st.markdown(f'#### Confidence: {round(class_5_confidence * 100, 1)}%')
        elif model_selection == "Random Forest":
            user_review = st.text_area("Enter your review here:", "I think this product is great! Well built and sturdy!")
            if st.button("Generate Prediction"):
                lem_review = new_column_lemmatizer(user_review)
                cv_review = new_count_vectorize_data(lem_review)
                [user_result] = rf_model.predict(cv_review)
                [[class_1_confidence, class_2_confidence, class_3_confidence, class_4_confidence, class_5_confidence]] = rf_model.predict_proba(cv_review)
                st.markdown('### Model Prediction')
                st.markdown(f'#### Class: {user_result}')
                if user_result == 1:
                    st.markdown(f'#### Confidence: {round(class_1_confidence * 100, 1)}%')
                if user_result == 2:
                    st.markdown(f'#### Confidence: {round(class_2_confidence * 100, 1)}%')
                if user_result == 3:
                    st.markdown(f'#### Confidence: {round(class_3_confidence * 100, 1)}%')
                if user_result == 4:
                    st.markdown(f'#### Confidence: {round(class_4_confidence * 100, 1)}%')
                if user_result == 5:
                    st.markdown(f'#### Confidence: {round(class_5_confidence * 100, 1)}%')
    elif model_type_selection == "Regression":
        model_selection = st.selectbox('Use the model below to compare modelling results:', ["Hist Gradient Boosting Regressor"])
        if model_selection == "Hist Gradient Boosting Regressor":
            user_review = st.text_area("Enter your review here:", "I think this product is great! Well built and sturdy!")
            if st.button("Generate Prediction"):
                lem_review = new_column_lemmatizer(user_review)
                cv_review = new_tfidf_vectorize_data(lem_review)
                cv_dense = cv_review.toarray()
                [user_result] = hgbr_model.predict(cv_dense)
                st.markdown('### Model Prediction')
                st.markdown(f'#### Class: {round(user_result, 2)}')
if page == pages[4]: 
    st.title('Conclusion and Next Steps')
    st.markdown("## Conclusion")
    st.markdown("Obviously the models in this report leave some accuracy to be desired. At best they match \
            or marginally improve upon the baseline target accuracy of 69%, at worst the models dramatically reduce the \
            accuracy below that target value.")
    st.markdown("These models can not be used to predict review ratings with a high degree of confidence. Extreme\
                values can be predicted with some degree of confidence, but nuance is easily lost in the models. More work is rewuired to create \
                a model that provides confidence in the results provided.")
    st.markdown("## Further")
    st.markdown("Combining preprocessing techniques and modelling techniques into a Deep Learning model is a sensible next step \
            to pursue. Additionally, since the input data is heavily imbalanced and the greatest prediction \
            improvements were seen by increasing the size of the dataset, it could be wise to expand and diversify the \
            dataset by sampling from other amazon categories to generate more distinction, or by using a big data\
            methodology to enable the models to be trained on datasets with millions of records.")
    st.markdown("""In the machine Learning Methodologies section two areas for improvement were identified:
* Improving Preprocessing, by implementing feature reduction
* Improving Modelling, by using more advanced modelling Techniques.""")
    st.markdown("## Continuing the Investigation")
    st.markdown("In light of the poor results, some additional work was undertaken as a contribution to the next steps of the project. \
                These investigations were heavily time constricted, so only the 'low hanging fruit' options were investigated.")
    st.markdown("### Improving Preprocessing")
    st.markdown
