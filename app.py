import streamlit as st
import pandas as pd
import joblib
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split # Added for train_test_split

# Define the preprocessing function
def simple_preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # lowercase
    t = text.lower()
    # remove URLs/emails
    t = re.sub(r"https?://\S+|\S+@\S+", " ", t)
    # remove non-word characters
    t = re.sub(r"[^\w\s]", " ", t)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Get the absolute path of the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Sidebar for Inputs ---
st.sidebar.title("Inputs")
dataset_path = st.sidebar.text_input("Dataset CSV", "data/sms_spam_no_header.csv")
label_col = st.sidebar.text_input("Label column", "0")
text_col = st.sidebar.text_input("Text column", "1")
models_dir = st.sidebar.text_input("Models dir", "models")
test_size = st.sidebar.slider("Test size", 0.10, 0.40, 0.20)
seed = st.sidebar.number_input("Seed", value=42)
decision_threshold = st.sidebar.slider("Decision threshold", 0.10, 0.90, 0.5)

# --- Main App ---
st.title("Spam/Ham Classifier — Phase 4 Visualizations")
st.write("Interactive dashboard for data distribution, token patterns, and model performance")

# Load data
try:
    df = pd.read_csv(dataset_path, header=None)
except FileNotFoundError:
    st.error(f"Dataset not found at: {dataset_path}")
    st.stop()

if df.empty:
    st.error(f"The dataset at {dataset_path} is empty or could not be read.")
    st.stop()

# Ensure the columns are named 0 and 1 as expected
if len(df.columns) < 2:
    st.error(f"The dataset at {dataset_path} does not have enough columns. Expected at least 2, got {len(df.columns)}.")
    st.stop()

# Rename columns to 0 and 1 if they are not already
if not all(col in df.columns for col in [0, 1]):
    df.columns = range(len(df.columns)) # Assign integer column names
    st.warning("Columns were not named 0 and 1. Renamed columns to integer indices.")

st.write("DEBUG: DataFrame loaded successfully.")
st.write("DEBUG: DataFrame head:", df.head())
st.write("DEBUG: DataFrame columns:", df.columns)

# Convert label_col and text_col to integers
try:
    label_col_int = int(label_col)
    text_col_int = int(text_col)
except ValueError:
    st.error("Label column and Text column must be integers when using a dataset without a header.")
    st.stop()

st.write(f"DEBUG: label_col_int: {label_col_int} (type: {type(label_col_int)})")
st.write(f"DEBUG: text_col_int: {text_col_int} (type: {type(text_col_int)})")

# --- Data Overview ---
st.header("Data Overview")

# Class distribution
st.subheader("Class distribution")
fig = px.bar(df[label_col_int].value_counts(), title="Class Distribution") # Use label_col_int
st.plotly_chart(fig)


# Token replacements
st.subheader("Token replacements in cleaned text (approximate)")
def count_token_replacements(text):
    counts = {
        "<URL>": len(re.findall(r"https?://\S+", text)),
        "<EMAIL>": len(re.findall(r"\S+@\S+", text)),
        "<PHONE>": 0,  # Placeholder, phone number regex can be complex
        "<NUM>": len(re.findall(r"\d+", text)),
    }
    return counts

total_counts = {"<URL>": 0, "<EMAIL>": 0, "<PHONE>": 0, "<NUM>": 0}
for text in df[text_col_int]: # Use text_col_int
    if isinstance(text, str):
        counts = count_token_replacements(text)
        for key in total_counts:
            total_counts[key] += counts[key]

st.write(pd.DataFrame.from_dict(total_counts, orient="index", columns=["count"]))


# --- Top Tokens by Class ---
st.header("Top Tokens by Class")
top_n = st.slider("Top-N tokens", 10, 40, 20)

# Vectorize text
tfidf = TfidfVectorizer(preprocessor=simple_preprocess, max_features=1000)
X = tfidf.fit_transform(df[text_col_int].astype(str)) # Use text_col_int
y = df[label_col_int].map({'ham': 0, 'spam': 1}) # Use label_col_int

# Get feature names
feature_names = tfidf.get_feature_names_out()

# Function to get top tokens
def get_top_n_tokens(class_label, n=10):
    class_indices = df[df[label_col_int] == class_label].index # Use label_col_int
    class_X = X[class_indices]
    class_scores = class_X.sum(axis=0)
    sorted_indices = class_scores.argsort()[0, ::-1]
    top_tokens = [feature_names[i] for i in sorted_indices[0, :n].tolist()[0]]
    top_scores = [class_scores[0, i] for i in sorted_indices[0, :n].tolist()[0]]
    return pd.DataFrame({"token": top_tokens, "score": top_scores})

col1, col2 = st.columns(2)

with col1:
    st.subheader("Class: ham")
    top_ham_tokens = get_top_n_tokens("ham", top_n)
    st.write(top_ham_tokens)


with col2:
    st.subheader("Class: spam")
    top_spam_tokens = get_top_n_tokens("spam", top_n)
    st.write(top_spam_tokens)


# --- Model Performance (Test) ---
st.header("Model Performance (Test)")

# Load the model bundle
model_path = os.path.join(script_dir, models_dir, "svm_baseline.joblib")
try:
    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    tfidf_model = model_bundle["tfidf"]

    # Split data (assuming the model was trained on a similar split)
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col_int].astype(str), y, test_size=test_size, random_state=seed # Use label_col_int and text_col_int
    )

    X_test_vec = tfidf_model.transform(X_test)
    y_pred = model.predict(X_test_vec)

    # Confusion matrix
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # Threshold sweep
    st.subheader("Threshold sweep (precision/recall/f1)")
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test_vec)[:, 1]
        thresholds = [i / 100 for i in range(10, 91)]
        results = []
        for t in thresholds:
            y_pred_t = (y_scores >= t).astype(int).map({1: "spam", 0: "ham"})
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_t, average="binary", pos_label="spam")
            results.append({"threshold": t, "precision": precision, "recall": recall, "f1": f1})

        results_df = pd.DataFrame(results)
        fig = px.line(results_df, x="threshold", y=["precision", "recall", "f1"], title="Precision, Recall, and F1 Score vs. Threshold")
        st.plotly_chart(fig)

    else:
        st.write("The loaded model does not support probability estimates (`predict_proba`), so the threshold sweep cannot be generated.")


except FileNotFoundError:
    st.warning(f"Model not found at: {model_path}. Some features will be disabled.")
    model = None


# --- Live Inference ---
st.header("Live Inference")
st.write("Enter a message to classify")

message_input = st.text_area("Message")

if st.button("Classify"):
    if message_input:
        if model:
            # Preprocess the input
            preprocessed_message = simple_preprocess(message_input)

            # Vectorize the message
            vectorized_message = tfidf_model.transform([preprocessed_message])

            # Predict
            prediction = model.predict(vectorized_message)

            # Display result
            result = "Spam" if prediction[0] == 1 else "Ham"
            st.write(f"**Prediction:** {result}")

            if result == "Spam":
                st.error("This message is classified as Spam.")
            else:
                st.success("This message is classified as Ham.")
        else:
            st.warning("Cannot classify message because no model is loaded.")
    else:
        st.write("Please enter a message to classify.")
