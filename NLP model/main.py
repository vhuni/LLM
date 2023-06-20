# import important modules
import numpy as np
import pandas as pd
import joblib 
import warnings

# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB # classifier 
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# text preprocessing modules
from string import punctuation 
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import re #regular expression
# Download dependency
for dependency in (
    "brown",
    "names",
    "wordnet",
    "averaged_perceptron_tagger",
    "universal_tagset",
):
    nltk.download(dependency)
    

warnings.filterwarnings("ignore")
# seeding
np.random.seed(123)


# load data
data = pd.read_csv("../data/labeledTrainData.tsv", sep='\t')

# show top five rows of data
data.head()

# check the shape of the data
data.shape

# check missing values in data
data.isnull().sum()


# evalute news sentiment distribution
data.sentiment.value_counts()


stop_words =  stopwords.words('english')
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text =  re.sub(r'http\S+',' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text) # remove numbers
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer() 
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
        # Return a list of words
    return(text)

#clean the review
data["cleaned_review"] = data["review"].apply(text_cleaning)

#split features and target from  data 
X = data["cleaned_review"]
y = data.sentiment.values

# split data into train and validate
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    shuffle=True,
    stratify=y,
)


# Create a classifier in pipeline
sentiment_classifier = Pipeline(steps=[
                               ('pre_processing',TfidfVectorizer(lowercase=False)),
                                 ('naive_bayes',MultinomialNB())
                                 ])

# train the sentiment classifier 
sentiment_classifier.fit(X_train,y_train)

# test model performance on valid data 
y_preds = sentiment_classifier.predict(X_valid)


accuracy_score(y_valid,y_preds)


#save model 
joblib.dump(sentiment_classifier, '../models/sentiment_model_pipeline.pkl')


# load the sentiment model
with open(
    join(dirname(realpath(__file__)), "models/sentiment_model_pipeline.pkl"), "rb"
) as f:
    model = joblib.load(f)


def predict_sentiment(review: str):
    """
    A simple function that receive a review content and predict the sentiment of the content.
    :param review:
    :return: prediction, probabilities
    """
    # clean the review
    cleaned_review = text_cleaning(review)
    
    # perform prediction
    prediction = model.predict([cleaned_review])
    output = int(prediction[0])
    probas = model.predict_proba([cleaned_review])
    output_probability = "{:.2f}".format(float(probas[:, output]))
    
    # output dictionary
    sentiments = {0: "Negative", 1: "Positive"}
    
    # show results
    result = {"prediction": sentiments[output], "Probability": output_probability}
    return result
    
    
results = predict_sentiment(review))
print(results["prediction"])
print(results["Probability"])


