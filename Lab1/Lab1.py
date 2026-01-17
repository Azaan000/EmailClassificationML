from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
import re


df = pd.read_csv('email.csv')

df['Category'] = df['Category'].astype(str).str.lower().str.strip()
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})

df = df.dropna(subset=['Category'])


def clean(text): #Function to clean the text data
    text = text.lower()
    text = re.sub(r'\W',' ',text) #removes special characters
    text = re.sub(r'\d',' ',text) #removes digits
    text = re.sub(r'\s+',' ',text) #removes extra spaces
    return text

#The clean function is applied to the 'Message' column
df['Message'] = df['Message'].apply(clean)

vectorizer = TfidfVectorizer(stop_words = 'english',max_features = 3000) #Creating the TF-IDF vectorizer with a maximum of 3000 features and English stop words removed
X = vectorizer.fit_transform(df['Message']) 
y = df['Category']

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
model = MultinomialNB()
model.fit(X_train,y_train)
pred = model.predict(X_test)
accuracy = accuracy_score(y_test,pred)
con = confusion_matrix(y_test, pred)
cr = classification_report(y_test, pred)

print("Accuracy: ", accuracy)
print("Confusion Matrix:\n", con)
print("Classification Report:\n", cr)

def predict_email(text): #Function to predict if an email is spam or ham
    text = clean(text)
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)
    return 'Spam' if prediction[0] == 1 else 'Ham'

print(predict_email("Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize."))
print(predict_email("Hi John, can we reschedule our meeting to next week?"))









