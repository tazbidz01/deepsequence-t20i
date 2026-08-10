import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_models():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "hf_commentary_labels.csv")
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Run fetch_hf_dataset.py first.")
        return
        
    print("Loading bootstrapped dataset...")
    df = pd.read_csv(data_path)
    print(f"Total labeled examples: {len(df)}")
    
    # We will train 3 separate models
    targets = ['line', 'length', 'shot']
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Fit a common vectorizer
    print("\nFitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_features = vectorizer.fit_transform(df['text'])
    
    # Save the vectorizer
    joblib.dump(vectorizer, os.path.join(models_dir, "nlp_vectorizer.pkl"))
    print("Saved nlp_vectorizer.pkl")
    
    for target in targets:
        print(f"\n--- Training Model for: {target.upper()} ---")
        
        # We only want to train on rows where the target is NOT 'Unknown'
        df_target = df[df[target] != 'Unknown'].copy()
        
        if len(df_target) < 50:
            print(f"Not enough data to train {target} model. Only {len(df_target)} rows.")
            continue
            
        print(f"Training on {len(df_target)} valid examples...")
        
        X = vectorizer.transform(df_target['text'])
        y = df_target[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        
        # Train Logistic Regression
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Validation Accuracy for {target}: {acc*100:.2f}%")
        
        # Save model
        model_path = os.path.join(models_dir, f"nlp_model_{target}.pkl")
        joblib.dump(clf, model_path)
        print(f"Saved {model_path}")
        
    print("\nAll NLP ML models trained and saved successfully!")

if __name__ == "__main__":
    train_models()
