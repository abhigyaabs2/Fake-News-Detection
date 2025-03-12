from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load trained model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")  # Load HTML page

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form["news"]
    news_vector = vectorizer.transform([news_text]).toarray()
    prediction = model.predict(news_vector)[0]
    result = "Real News" if prediction == 1 else "Fake News"
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
