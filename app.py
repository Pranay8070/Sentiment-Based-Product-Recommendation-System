import nltk_setup  # Ensure NLTK data is set up before importing model
from flask import Flask, render_template, request
from model import SentimentRecommenderModel
import pandas as pd
import os

app = Flask(__name__)

# Load the model once at startup
model = SentimentRecommenderModel()

@app.route("/", methods=["GET", "POST"])
def home():
    # Default empty DataFrame (VERY IMPORTANT for Jinja)
    recommendations = pd.DataFrame(
        columns=["name", "brand", "manufacturer", "pos_sentiment_percent"]
    )

    user_name = ""
    error_message = ""

    # 🔹 Sample usernames for dropdown
    sample_users = [
        "samantha",
        "john_doe",
        "michael",
        "emma",
        "robert"
    ]

    if request.method == "POST":
        user_name = request.form.get("username", "").strip()

        if user_name:
            recommendations = model.getSentimentRecommendations(user_name)

            # Safety check: always ensure DataFrame
            if not isinstance(recommendations, pd.DataFrame):
                recommendations = pd.DataFrame()

            if recommendations.empty:
                error_message = (
                    f"No recommendations found for user '{user_name}'. "
                    "Please try another username."
                )
        else:
            error_message = "Please enter a username."

    return render_template(
        "index.html",
        recommendations=recommendations,
        user_name=user_name,
        error_message=error_message,
        sample_users=sample_users  # 🔹 passed to HTML dropdown
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
