import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# PAGE CONFIG (must be first Streamlit command)
st.set_page_config(page_title="Schizophrenia Detection System", layout="wide")

# TITLE
st.title("EEG-Based Schizophrenia Detection System")
st.markdown("AI Model for detecting Schizophrenia from EEG features")

# LOAD MODEL (cached so it loads only once)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("schizophrenia_model.h5", compile=False)
    return model

model = load_model()

model = load_model()

# SIDEBAR
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Convert dataset to numpy
    X = df.values
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # PREDICTION
    predictions = model.predict(X)

    probs = predictions.flatten()

    labels = []

    for p in probs:
        if p > 0.5:
            labels.append("Schizophrenia (SZ)")
        else:
            labels.append("Healthy Control (HC)")

    df["Prediction Probability"] = probs
    df["Predicted Class"] = labels

    st.subheader("Prediction Results")
    st.dataframe(df)

    # COUNT RESULTS
    sz_count = labels.count("Schizophrenia (SZ)")
    hc_count = labels.count("Healthy Control (HC)")

    st.subheader("Prediction Summary")

    col1, col2 = st.columns(2)

    col1.metric("Schizophrenia Cases", sz_count)
    col2.metric("Healthy Controls", hc_count)

    # BAR GRAPH
    fig, ax = plt.subplots()

    classes = ["Schizophrenia", "Healthy"]
    values = [sz_count, hc_count]

    ax.bar(classes, values)

    ax.set_title("Prediction Distribution")

    st.pyplot(fig)

else:
    st.info("Upload a CSV file to start prediction.")