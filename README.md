# 🛍️ Product Review Sentiment Analysis using Word Embeddings

## 📌 Problem Statement

### 🧭 Business Context

In today’s e-commerce environment, customer reviews significantly influence buyer decisions. Understanding the sentiment expressed in these reviews can help businesses enhance customer experience, reduce churn, and improve product offerings.

This project uses natural language processing (NLP) techniques and pre-trained word embeddings to analyze product review sentiment (Positive, Neutral, or Negative).

---

## 📂 Dataset

- **File**: `Product_Reviews.csv`
- **Fields**:
  - `Review_Text`: Text of the review
  - `Sentiment`: Labeled sentiment (Positive, Negative, Neutral)

---

## 🛠️ Tools & Technologies

- Python 3.x
- pandas, NumPy
- nltk, gensim, keras
- scikit-learn
- matplotlib, seaborn

---

## 🔁 Workflow

```python
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

# 2. Load Dataset
df = pd.read_csv('Product_Reviews.csv')

# 3. Preprocessing
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [word for word in text.split() if word not in stopwords]
    return ' '.join(words)

df['clean_text'] = df['Review_Text'].apply(clean_text)

# 4. Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(sequences, maxlen=100)

# 5. Prepare Labels
sentiment_map = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
y = df['Sentiment'].map(sentiment_map)

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Load Pretrained Word Embeddings
word_vectors = api.load("glove-wiki-gigaword-100")
embedding_matrix = np.zeros((5000, 100))
for word, i in tokenizer.word_index.items():
    if i < 5000 and word in word_vectors:
        embedding_matrix[i] = word_vectors[word]

# 8. Model Building
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=100, weights=[embedding_matrix], input_length=100, trainable=False))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 9. Model Training
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# 10. Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
```

---

## 📈 Key Outcomes

- Leveraged pre-trained GloVe embeddings for effective semantic representation.
- Achieved good classification accuracy with LSTM-based neural architecture.
- Demonstrated scalable approach to handle customer feedback in real-world settings.

---

## 📁 Repository Structure

```
├── Case_Study_Product_Review_Sentiment_Analysis_Word_Embeddings.ipynb
├── Product_Reviews.csv
├── README.md
```

---

## 👨‍💻 Author

**Suhaib Khalid**  
NLP & Deep Learning Practitioner 
