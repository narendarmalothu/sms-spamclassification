import streamlit as st
import pickle
import nltk
import re
import string
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
nltk.download('punkt_tab')
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)

    # removing special char
    y=[]
    for i in text:
        y.extend(re.findall('[a-zA-Z0-9]+',i))
        # if i.isalnum():
        #     y.append(i)
    text=y.copy()
    y.clear()
    for i in text:
       if i not in  stopwords.words('english') and i not in string.punctuation: # punctuation not necessary here
           y.append(i)
    text=y.copy()
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return ' '.join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('Email/SMS spam Classifier')

input_sms=st.text_area('Enter the Message')
if st.button('Predict'):
    # preprocess
    transformed_sms=transform_text(input_sms)
    # vectorize
    vector_input=tfidf.transform([transformed_sms])
    # predict
    result=model.predict(vector_input)[0]
    # display
    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')