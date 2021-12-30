import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from senticnet.senticnet import SenticNet

url = "https://www.amazon.in/Noise-Colorfit-Pro-Touch-Control/product-reviews/B07YY1BY5B/ref=cm_cr_getr_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=2"

from Scraper_Main import amazon

reviews_list = amazon(url, 10)

len(reviews_list)
raw_reviews = pd.DataFrame({'reviews': reviews_list})
raw_reviews.shape


def text_clean():
    for i in range(0, len(raw_reviews.reviews)):
        raw_reviews['reviews'].iloc[i] = re.sub("RT @[\w_]+: ", "", raw_reviews['reviews'].iloc[i])
        raw_reviews['reviews'].iloc[i] = re.sub("<.*?>", "", raw_reviews['reviews'].iloc[i])
        raw_reviews['reviews'].iloc[i] = re.sub(r'[^\x00-\x7F]+', '', raw_reviews['reviews'].iloc[i])
        raw_reviews['reviews'].iloc[i] = re.sub(' +', ' ', raw_reviews['reviews'].iloc[i])
        raw_reviews['reviews'].iloc[i] = raw_reviews['reviews'].iloc[i].lower()
        raw_reviews['reviews'].iloc[i] = re.sub("[^\w\s]", "", raw_reviews['reviews'].iloc[i])
        raw_reviews['reviews'].iloc[i] = re.sub('[^0-9a-zA-Z ]+', "", raw_reviews['reviews'].iloc[i])
    return raw_reviews


raw_reviews.head(10)
clean_reviews = text_clean()

len(clean_reviews)

stopwords_file = open("stopwords.txt")
stopwords_user = set(stopwords_file.read().split())
from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')
stopwords_english = set(stopwords.words('english'))
stopwords = list(stopwords_user.union(stopwords_english))
len(stopwords)

clean_reviews['without_stopwords'] = clean_reviews['reviews'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

clean_reviews_final = pd.DataFrame(clean_reviews.without_stopwords)
clean_reviews_final.head(5)
len(clean_reviews_final)

import nltk
nltk.download('punkt')
for j in range(1, len(clean_reviews_final)):
    if len(word_tokenize(str(clean_reviews_final.without_stopwords[j]))) < 1:
        clean_reviews_final = clean_reviews_final.drop([j])

len(clean_reviews_final)

clean_reviews_series = clean_reviews_final.without_stopwords

vectorizerng = CountVectorizer(ngram_range=(1, 2), min_df=0.01)

document_term_matrix_ng = vectorizerng.fit_transform(clean_reviews_series)

document_term_matrix_ng = pd.DataFrame(document_term_matrix_ng.toarray(), columns=vectorizerng.get_feature_names_out())

document_term_matrix_ng.shape

document_term_matrix_ng.head(10)

words = dict(document_term_matrix_ng.apply(sum, axis=0))
wordcloud = WordCloud(max_font_size=40, max_words=50, background_color="white").fit_words(words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

vectorizer = CountVectorizer()

document_term_matrix = vectorizer.fit_transform(clean_reviews_series)

document_term_matrix = pd.DataFrame(document_term_matrix.toarray(), columns=vectorizer.get_feature_names_out())

document_term_matrix.shape

words = dict(document_term_matrix.apply(sum, axis=0))
wordcloud = WordCloud(max_font_size=40, max_words=50, background_color="white").fit_words(words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

vectorizeridf = TfidfVectorizer()

document_term_matrix_idf = vectorizeridf.fit_transform(clean_reviews_series)

document_term_matrix_idf = pd.DataFrame(document_term_matrix_idf.toarray(), columns=vectorizeridf.get_feature_names_out())

document_term_matrix_idf.shape

words = dict(document_term_matrix_idf.apply(sum, axis=0))
wordcloud = WordCloud(max_font_size=40, max_words=50, background_color="white").fit_words(words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
