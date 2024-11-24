import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Preprocess symptoms text
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load dataset
df = pd.read_csv(r"dataset\dataset.csv")

# Preprocess symptoms
df['processed_symptoms'] = df['symptoms'].apply(preprocess_text)

# Vectorize symptoms using TF-IDF
vectorizer = TfidfVectorizer()
symptom_vectors = vectorizer.fit_transform(df['processed_symptoms'])

# Convert to dense array for distance calculation
symptom_dense = symptom_vectors.toarray()

# Perform hierarchical clustering with single-link (complete) criteria
linkage_matrix = linkage(symptom_dense, method='single')

# Plot dendrogram
plt.figure(figsize=(15, 10))
dendrogram(
    linkage_matrix,
    labels=df['disease'].tolist(),
    leaf_rotation=90,
    leaf_font_size=8,
    show_leaf_counts=True
)
plt.title('Hierarchical Clustering of Diseases based on Symptom Similarity', fontsize=16)
plt.xlabel('Diseases', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.tight_layout()
plt.show()
