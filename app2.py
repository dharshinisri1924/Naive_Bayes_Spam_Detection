import streamlit as st
import pickle

#st.write("This is app2.py")
# Load Naive Bayes model
with open("NB_model.pkl", "rb") as file:
    model = pickle.load(file)


# Load vectorizer
with open("vector.pkl", "rb") as file:
    vectorizer = pickle.load(file)

st.set_page_config(
    page_title="Spam Detector",
    layout="centered"
)

st.title("📧 Email Spam Detection using Naive Bayes")

email = st.text_area(
    "Enter Email",
    height=220,
    placeholder="Paste your email here..."
)

col1, col2 = st.columns(2)

with col1:
    predict = st.button("Predict")

with col2:
    clear = st.button("Clear")

if predict:
    if email.strip() == "":
        st.warning("Please enter an email.")
    else:
        vector = vectorizer.transform([email])
        prediction = model.predict(vector)[0]

        st.divider()

        if prediction == 1:
            st.error("🚨 This Email is SPAM")
        else:
            st.success("✅ This Email is NOT SPAM")

if clear:
    st.rerun()