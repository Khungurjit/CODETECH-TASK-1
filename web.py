import pickle as pk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
model=pk.load(open('model.pkl','rb'))
scaler=pk.load(open('scaler.pkl','rb'))
st.title("Movie Review Sentiment Analysis")
review=st.text_input("Enter your opinion")
if st.button("predict"):
    review_scale=scaler.transform([review]).toarray()
    output=model.predict(review_scale)
    if output[0]==0:
        st.write("negative review")
    else:
        st.write("postive review")