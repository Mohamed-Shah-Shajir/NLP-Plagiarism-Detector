# NLP-Based Plagiarism Detection and Text Similarity Analysis

## Project Overview

This project implements an NLP-based plagiarism detection and text similarity analysis system using TF-IDF vectorization and cosine similarity.

The system compares two text documents and calculates a similarity score. Based on the score, the texts are categorized as Low Similarity, Moderate Similarity, or High Similarity.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TF-IDF
- Cosine Similarity

## Dataset

The project uses the Text Similarity Dataset obtained from Kaggle.

The dataset contains pairs of text documents with the following attributes:

- Unique_ID
- text1
- text2

## Methodology

Text Dataset  
↓  
Text Preprocessing  
↓  
TF-IDF Vectorization  
↓  
Cosine Similarity  
↓  
Similarity Classification  
↓  
Plagiarism Detection

## Similarity Categories

| Score | Category |
|---|---|
| 0.50 and above | High Similarity |
| 0.20 - 0.49 | Moderate Similarity |
| Below 0.20 | Low Similarity |

## Project Results

The system successfully calculates similarity scores between text pairs and identifies highly similar documents.

A pair with a similarity score of 1.0 indicates that the texts are effectively identical after preprocessing.

## Project Structure

```text
NLP-Plagiarism-Detector/
│
├── src/
├── dataset/
├── results/
├── screenshots/
├── report/
├── requirements.txt
├── README.md
└── .gitignore