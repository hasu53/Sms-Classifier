import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset with the correct encoding
df = pd.read_csv("spam.csv", encoding='ISO-8859-1', sep=',', usecols=[0, 1], names=['label', 'message'], header=0)

# Display the first few rows of the dataset
print(df.head())

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Build a pipeline for text processing and classification
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),       # Convert text to a matrix of token counts
    ('tfidf', TfidfTransformer()),           # Transform counts to a normalized tf-idf representation
    ('classifier', MultinomialNB())          # Naive Bayes classifier
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

print('\nClassification Report:')
print(classification_report(y_test, y_pred))

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Function to classify new messages
def classify_message(message):
    prediction = pipeline.predict([message])[0]
    return prediction

# Test the function with a new message
test_message = "Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/12345 to claim now."
print(f'\nTest Message: "{test_message}"')
print(f'Classification: {classify_message(test_message)}')

