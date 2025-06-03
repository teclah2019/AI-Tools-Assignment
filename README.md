# AI-Tools-Assignment
1. Short Answer Questions
Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?

TensorFlow uses static computation graphs by default (although eager execution is now available), is widely used in production environments, and has strong deployment support (e.g., TensorFlow Serving, TensorFlow Lite, TensorFlow.js).

PyTorch uses dynamic computation graphs (define-by-run), making it more intuitive and flexible for research and experimentation.

Choose TensorFlow if you want strong deployment options and scalability for production.

Choose PyTorch if you prioritize ease of debugging, research flexibility, or prefer Pythonic coding style.

Q2: Describe two use cases for Jupyter Notebooks in AI development.

Data exploration & visualization: Interactive cells allow data scientists to load datasets, run visualizations (matplotlib, seaborn), and tweak preprocessing steps quickly.

Model prototyping & experimentation: Developers can iteratively write and test model code, visualize training progress, and document findings within the notebook.

Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?

spaCy provides pretrained, optimized pipelines for tokenization, part-of-speech tagging, dependency parsing, and named entity recognition (NER).

It handles complex language structures reliably and efficiently, whereas basic string operations are error-prone, limited, and lack linguistic understanding.

spaCy offers easy integration with machine learning workflows and supports customizable pipelines.

2. Comparative Analysis: Scikit-learn vs TensorFlow
Aspect	Scikit-learn	TensorFlow
Target Applications	Classical ML (e.g., SVM, DT, KNN)	Deep learning (CNNs, RNNs, etc.)
Ease of Use	Beginner-friendly, simple API	Steeper learning curve, more complex
Community Support	Mature, large community with many resources	Growing rapidly, large research and production community

Part 2: Practical Implementation (50%)
Task 1: Classical ML with Scikit-learn
Dataset: Iris Species Dataset

Steps:

python
Copy
Edit
# Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load Iris data
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess (no missing values in Iris, but encode labels if needed)
# Labels already numeric

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (macro):", precision_score(y_test, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))
Task 2: Deep Learning with TensorFlow
Dataset: MNIST Handwritten Digits

Steps:

python
Copy
Edit
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load MNIST
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize and reshape
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=5, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

# Visualize predictions on 5 images
import numpy as np

predictions = model.predict(X_test[:5])
fig, axs = plt.subplots(1, 5, figsize=(15,3))
for i in range(5):
    axs[i].imshow(X_test[i].reshape(28,28), cmap='gray')
    axs[i].set_title(f"Pred: {np.argmax(predictions[i])}\nTrue: {y_test[i]}")
    axs[i].axis('off')
plt.show()
Task 3: NLP with spaCy
Text Data: Amazon Product Reviews (example text used here)

python
Copy
Edit
import spacy
from spacy import displacy

# Load pre-trained English model
nlp = spacy.load("en_core_web_sm")

# Sample review text
reviews = [
    "I love the new Apple iPhone! The camera is fantastic.",
    "The Samsung Galaxy has a great display but battery life is short.",
    "Bought the Bose headphones, sound quality is amazing."
]

for review in reviews:
    doc = nlp(review)
    print(f"Review: {review}")
    # Extract named entities
    for ent in doc.ents:
        print(f" - Entity: {ent.text}, Label: {ent.label_}")
    # Simple rule-based sentiment (very basic)
    sentiment = "Positive" if "love" in review or "great" in review or "amazing" in review else "Negative/Neutral"
    print(f"Sentiment: {sentiment}\n")
Part 3: Ethics & Optimization (10%)
1. Ethical Considerations
Potential Biases:

MNIST: Limited to handwritten digits from certain demographics, may fail on diverse handwriting styles.

Amazon Reviews: Reviews may contain biased opinions or lack representation of all user groups.

Mitigation:

Use TensorFlow Fairness Indicators to detect bias in model predictions across subgroups.

For NLP, use spaCy’s rule-based systems combined with diverse training data to avoid biased entity recognition.

Continually audit data and models for fairness.

2. Troubleshooting Challenge
Example: Fix dimension mismatch in TensorFlow model by ensuring input shape matches data.

Bonus Task (Extra 10%)
Use Streamlit to deploy your MNIST classifier:

python
Copy
Edit
import streamlit as st
from PIL import Image
import numpy as np

st.title("MNIST Digit Classifier")

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28,28))
    img_array = np.array(image)/255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    st.write(f"Predicted digit: {np.argmax(prediction)}")
    st.image(image, caption='Uploaded Image', use_column_width=True)
