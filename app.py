import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean data
df = pd.read_csv("Roman Urdu DataSet.csv")
df = df.rename(columns={df.columns[0]: "Review", df.columns[1]: "Sentiment"})
df = df[['Review', 'Sentiment']].dropna()

# Prepare data
X = df['Review'].astype(str)
y = df['Sentiment'].astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

# Sidebar - Navigation
st.sidebar.title("RomanSentiment")
option = st.sidebar.selectbox("Navigate", ["Home", "Dataset", "Evaluation", "Predict"])

# Sidebar - User input
if option == "Predict":
    st.sidebar.subheader("Enter a Roman Urdu sentence")
    user_input = st.sidebar.text_area("Your Text:")
else:
    user_input = None

# Page Content
st.title("📊 Roman Urdu Sentiment Analyzer")

if option == "Home":
    st.markdown("### 🤖 A simple machine learning app to predict Roman Urdu sentence sentiment.")
    st.markdown("Built using **Naive Bayes**, **TF-IDF**, and **Streamlit**.")
    st.markdown("Navigate to different sections using the left sidebar.")

elif option == "Dataset":
    st.header("🔍 Dataset Preview")
    st.write(df.head())

    st.header("📈 Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Sentiment', palette="Set2", ax=ax)
    st.pyplot(fig)

elif option == "Evaluation":
    st.header("📊 Model Evaluation")
    st.success(f"✅ Accuracy: {accuracy * 100:.2f}%")

    st.subheader("📄 Classification Report")
    st.dataframe(pd.DataFrame(class_report).transpose())

    st.subheader("🔁 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

elif option == "Predict":
    st.header("🔮 Predict Sentiment")

    if user_input:
        user_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_vec)[0]
        st.success(f"**Predicted Sentiment:** {prediction}")
    else:
        st.info("Enter text in the sidebar to get prediction.")
