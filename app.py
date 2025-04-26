from flask import Flask, render_template, request
import pickle
import string
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the saved model and vectorizer
model = pickle.load(open('spam_classifier_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    cleaned_message = clean_text(message)
    transformed_message = vectorizer.transform([cleaned_message])
    prediction = model.predict(transformed_message)[0]

    if prediction == 1:
        result = "✅ This message is NOT Spam (Ham)!"
        color = "success"
    else:
        result = "🚫 Warning! This message is Spam!"
        color = "danger"

    return render_template('index.html', prediction_text=result, color=color)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
