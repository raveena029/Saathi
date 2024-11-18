# Saathi: Health and Crime Analysis System

## Overview
Saathi is a comprehensive Streamlit application that provides two key analytical features:
1. Women Crime Report Analysis
2. Disease Classification Based on Symptoms

## Features

### Women Crime Report
- Interactive crime data analysis for Indian states
- K-means clustering of crime patterns
- State-wise crime ranking
- Filtering by year, state, and crime type
- Visualization of crime pattern clusters
- Classification metrics computation

### Disease Classification
- Symptom-based disease prediction
- Spelling correction for input symptoms
- K-Nearest Neighbors (KNN) classification
- Model performance metrics
- Detailed disease information including:
  - Predicted disease
  - Recommended medication
  - Suggested doctor specialty
  - Risk level
  - Similarity score

## Technologies Used
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Jellyfish (for spelling correction)

## Prerequisites
- Python 3.7+
- Libraries listed in requirements.txt

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run final.py`

## Data Sources
- CrimesOnWomenData.csv (Crime analysis)
- dataset/dataset.csv (Disease classification)

## Configuration
- Customize file paths in the script if needed
- Adjust model parameters for fine-tuning

