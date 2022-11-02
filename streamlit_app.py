# -*- coding: utf-8 -*-


import streamlit as st 
from sentence_transformers import SentenceTransformer, util
from PIL import Image

def similarity_scorer(text1,text2):
     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
     #Compute embedding for 2 sentences
     vector_1= model.encode(text1, convert_to_tensor=True)
     vector_2 = model.encode(text2, convert_to_tensor=True)
     #cosine similarity
     score=float(util.pytorch_cos_sim(vector_1, vector_2))
     return score


def main():
    st.title("Similarity Checker")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Sentence Similarity Scorer </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    text1 = st.text_input("text1")
    text2 = st.text_input("text2")
    
    similarity_score=0
    if st.button("Calculate"):
        similarity_score=similarity_scorer(text1,text2)
    st.success(similarity_score)
    if st.button("About"):
        st.text("Made for Precily")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
