import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# 1. Training Corpus (Representative ham and spam samples)
training_data = [
    # Legitimate (Ham) Messages
    ("Hey Gaurav, are we still meeting tomorrow for the project review?", "ham"),
    ("Can you please send me the updated meeting notes from today's sync?", "ham"),
    ("The code deployment to the staging server was successful.", "ham"),
    ("Let's review the pull request before pushing to the main branch.", "ham"),
    ("Good morning, hope you have a productive week ahead.", "ham"),
    ("Please find attached the quarterly invoice for your records.", "ham"),
    ("Don't forget to push your branch to GitHub before logging off.", "ham"),
    ("What time is our sync scheduled with the client tomorrow?", "ham"),
    ("The database migration script completed with zero errors.", "ham"),
    ("Thanks for your quick response on the API documentation.", "ham"),
    ("Can we reschedule our call to 3:30 PM this afternoon?", "ham"),
    ("I have updated the Jira ticket with the latest latency benchmarks.", "ham"),
    
    # Spam / Malicious / Phishing Messages
    ("URGENT! You have won a $1,000 cash voucher. Click the link now to claim your prize!", "spam"),
    ("Congratulations! You are the selected winner of a brand new iPhone. Reply WIN to claim.", "spam"),
    ("Your bank account has been locked due to suspicious activity. Verify credentials immediately at http://security-verify.test", "spam"),
    ("Claim your guaranteed free vacation package now! Call 1-800-FREE-TRIP today only.", "spam"),
    ("Exclusive crypto investment opportunity! Earn 500% ROI in 24 hours. Sign up here.", "spam"),
    ("WINNER! You have won an instant $2,000 cash voucher. Click to claim your award immediately.", "spam"),
    ("URGENT: Your credit card payment failed. Update your card number at http://billing-update.test", "spam"),
    ("Free entry into our weekly £10,000 prize draw! Text ENTRY to 88088 now.", "spam"),
    ("Double your Bitcoin in 48 hours guaranteed. Deposit to our official wallet now.", "spam"),
    ("Important security alert: unauthorized login detected. Reset your password at http://auth-reset.test", "spam"),
    ("Congratulations, you have been selected for a pre-approved loan with 0% interest! Apply today.", "spam"),
    ("Final notice: your package is on hold at our distribution facility. Pay delivery fee to release.", "spam")
]

def train_and_export():
    # Ensure destination folder exists
    os.makedirs("model_files", exist_ok=True)

    # Separate texts and target labels
    texts = [item[0] for item in training_data]
    labels = [1 if item[1] == "spam" else 0 for item in training_data]

    print(">>> Vectorizing training dataset with TF-IDF...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(texts)

    print(">>> Training Multinomial Naive Bayes Classifier...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X, labels)

    # In-sample evaluation
    preds = model.predict(X)
    acc = accuracy_score(labels, preds)
    print(f"\nModel Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=["Ham", "Spam"]))

    # Serialize artifacts for FastAPI inference
    model_path = "model_files/spam_model.joblib"
    vectorizer_path = "model_files/vectorizer.joblib"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"\n Artifacts successfully exported:")
    print(f"  - Model: {model_path}")
    print(f"  - Vectorizer: {vectorizer_path}")

if __name__ == "__main__":
    train_and_export()