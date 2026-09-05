# ============================================================
# NLP-BASED PLAGIARISM DETECTION AND TEXT SIMILARITY ANALYSIS
# Using TF-IDF and Cosine Similarity
# ============================================================

import os
import re
import string

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


print("=" * 80)
print("NLP PLAGIARISM DETECTION SYSTEM")
print("=" * 80)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

DATASET_PATH = "dataset/Text_Similarity_Dataset.csv"

PROCESSED_DATASET_PATH = (
    "dataset/processed_text_similarity_dataset.csv"
)

RESULTS_DIR = "results"

SIMILARITY_RESULTS_PATH = (
    "results/similarity_results.csv"
)

TOP_10_RESULTS_PATH = (
    "results/top_10_similar_pairs.csv"
)

LEAST_5_RESULTS_PATH = (
    "results/least_5_similar_pairs.csv"
)

SIMILARITY_GRAPH_PATH = (
    "results/similarity_distribution.png"
)

CATEGORY_GRAPH_PATH = (
    "results/similarity_category_distribution.png"
)


# Create required folders
os.makedirs("dataset", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("\nProject directories checked successfully!")


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 80)
print("1. LOADING DATASET")
print("=" * 80)

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")

print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("2. BASIC DATASET INFORMATION")
print("=" * 80)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nUnique IDs:")
print(df["Unique_ID"].nunique())


# ============================================================
# 4. REMOVE INCOMPLETE TEXT PAIRS
# ============================================================

print("\n" + "=" * 80)
print("3. REMOVING INCOMPLETE TEXT PAIRS")
print("=" * 80)

initial_rows = len(df)

df = df.dropna(
    subset=["text1", "text2"]
).copy()

removed_rows = initial_rows - len(df)

print("\nRows removed:", removed_rows)
print("Rows remaining:", len(df))


# ============================================================
# 5. SAMPLE TEXT PAIR
# ============================================================

print("\n" + "=" * 80)
print("4. SAMPLE TEXT PAIR")
print("=" * 80)

print("\nTEXT 1:")
print(df["text1"].iloc[0])

print("\nTEXT 2:")
print(df["text2"].iloc[0])


# ============================================================
# 6. TEXT LENGTH ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("5. TEXT LENGTH ANALYSIS")
print("=" * 80)


df["text1_word_count"] = (
    df["text1"]
    .str.split()
    .str.len()
)

df["text2_word_count"] = (
    df["text2"]
    .str.split()
    .str.len()
)


print("\nText length statistics:")

print(
    df[
        [
            "text1_word_count",
            "text2_word_count"
        ]
    ].describe()
)


# ============================================================
# 7. NLP PREPROCESSING
# ============================================================

print("\n" + "=" * 80)
print("6. NLP PREPROCESSING")
print("=" * 80)


def preprocess_text(text):
    """
    Clean and preprocess text.

    Steps:
    1. Convert text to lowercase
    2. Remove URLs
    3. Remove punctuation
    4. Remove stopwords
    5. Remove extra whitespace
    """

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Split into words
    words = text.split()

    # Remove English stopwords
    words = [
        word
        for word in words
        if word not in ENGLISH_STOP_WORDS
    ]

    # Join words
    text = " ".join(words)

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# 8. APPLY NLP PREPROCESSING
# ============================================================

df["clean_text1"] = (
    df["text1"]
    .apply(preprocess_text)
)

df["clean_text2"] = (
    df["text2"]
    .apply(preprocess_text)
)


print("\nPreprocessing completed successfully!")


# ============================================================
# 9. SHOW PREPROCESSING EXAMPLE
# ============================================================

print("\n" + "=" * 80)
print("TEXT PREPROCESSING COMPARISON")
print("=" * 80)

print("\nOriginal Text 1:")
print(df["text1"].iloc[0])

print("\nCleaned Text 1:")
print(df["clean_text1"].iloc[0])


# ============================================================
# 10. CHECK EMPTY DOCUMENTS
# ============================================================

empty_text1 = (
    df["clean_text1"] == ""
).sum()

empty_text2 = (
    df["clean_text2"] == ""
).sum()


print("\nEmpty cleaned Text 1:", empty_text1)
print("Empty cleaned Text 2:", empty_text2)


# ============================================================
# 11. WORD COUNT AFTER PREPROCESSING
# ============================================================

df["clean_text1_word_count"] = (
    df["clean_text1"]
    .str.split()
    .str.len()
)

df["clean_text2_word_count"] = (
    df["clean_text2"]
    .str.split()
    .str.len()
)


print("\nAverage word counts:")

print(
    "Text 1 before preprocessing:",
    round(
        df["text1_word_count"].mean(),
        2
    )
)

print(
    "Text 1 after preprocessing:",
    round(
        df["clean_text1_word_count"].mean(),
        2
    )
)

print(
    "Text 2 before preprocessing:",
    round(
        df["text2_word_count"].mean(),
        2
    )
)

print(
    "Text 2 after preprocessing:",
    round(
        df["clean_text2_word_count"].mean(),
        2
    )
)


# ============================================================
# 12. SAVE PROCESSED DATASET
# ============================================================

print("\n" + "=" * 80)
print("7. SAVING PROCESSED DATASET")
print("=" * 80)


df.to_csv(
    PROCESSED_DATASET_PATH,
    index=False
)


print("\nProcessed dataset saved successfully!")

print(
    "Location:",
    PROCESSED_DATASET_PATH
)


# ============================================================
# 13. TF-IDF VECTORIZATION
# ============================================================

print("\n" + "=" * 80)
print("8. TF-IDF VECTORIZATION")
print("=" * 80)


# Combine both text columns
# so that the same vocabulary is used.

all_text = pd.concat(
    [
        df["clean_text1"],
        df["clean_text2"]
    ],
    ignore_index=True
)


print(
    "\nTotal documents for TF-IDF:",
    len(all_text)
)


# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()


print(
    "TF-IDF vectorizer created successfully!"
)


# Learn vocabulary and IDF values
tfidf_vectorizer.fit(
    all_text
)


print(
    "TF-IDF vectorizer fitted successfully!"
)


# Transform Text 1
tfidf_text1 = (
    tfidf_vectorizer.transform(
        df["clean_text1"]
    )
)


# Transform Text 2
tfidf_text2 = (
    tfidf_vectorizer.transform(
        df["clean_text2"]
    )
)


print("\nTF-IDF Vectorization completed!")

print(
    "Text 1 TF-IDF shape:",
    tfidf_text1.shape
)

print(
    "Text 2 TF-IDF shape:",
    tfidf_text2.shape
)


# ============================================================
# 14. VOCABULARY ANALYSIS
# ============================================================

feature_names = (
    tfidf_vectorizer
    .get_feature_names_out()
)


print(
    "\nNumber of vocabulary terms:",
    len(feature_names)
)

print("\nFirst 30 vocabulary terms:")
print(feature_names[:30])


# ============================================================
# 15. FIRST TF-IDF VECTOR
# ============================================================

first_vector = tfidf_text1[0]


print("\nFirst TF-IDF vector:")
print(first_vector)

print(
    "\nNumber of non-zero values:",
    first_vector.nnz
)


# ============================================================
# 16. TOP 10 TF-IDF WORDS
# ============================================================

scores = (
    tfidf_text1[0]
    .toarray()
    .flatten()
)


tfidf_scores = pd.DataFrame(
    {
        "word": feature_names,
        "tfidf_score": scores
    }
)


tfidf_scores = (
    tfidf_scores
    .sort_values(
        by="tfidf_score",
        ascending=False
    )
)


print(
    "\nTop 10 important words in the first document:"
)

print(
    tfidf_scores.head(10)
)


# ============================================================
# 17. COSINE SIMILARITY
# ============================================================

print("\n" + "=" * 80)
print("9. COSINE SIMILARITY")
print("=" * 80)


# Calculate similarity between
# corresponding Text 1 and Text 2.

pair_similarity_scores = (
    tfidf_text1
    .multiply(tfidf_text2)
    .sum(axis=1)
    .A1
)


# Add similarity score
df["similarity_score"] = (
    pair_similarity_scores
)


print(
    "\nCosine similarity calculated successfully!"
)


print("\nFirst 10 similarity scores:")

print(
    df[
        [
            "Unique_ID",
            "similarity_score"
        ]
    ].head(10)
)


# ============================================================
# 18. SIMILARITY SCORE STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("10. SIMILARITY SCORE ANALYSIS")
print("=" * 80)


print("\nSimilarity Score Statistics:")

print(
    df["similarity_score"].describe()
)


# ============================================================
# 19. SIMILARITY DISTRIBUTION GRAPH
# ============================================================

print("\n" + "=" * 80)
print("11. SIMILARITY DISTRIBUTION GRAPH")
print("=" * 80)


plt.figure(
    figsize=(10, 5)
)

plt.hist(
    df["similarity_score"],
    bins=30
)

plt.title(
    "Distribution of Cosine Similarity Scores"
)

plt.xlabel(
    "Cosine Similarity Score"
)

plt.ylabel(
    "Number of Text Pairs"
)

plt.tight_layout()

plt.savefig(
    SIMILARITY_GRAPH_PATH
)

plt.show()


print(
    "\nSimilarity distribution graph saved:"
)

print(
    SIMILARITY_GRAPH_PATH
)


# ============================================================
# 20. TOP 10 MOST SIMILAR TEXT PAIRS
# ============================================================

print("\n" + "=" * 80)
print("12. TOP 10 MOST SIMILAR TEXT PAIRS")
print("=" * 80)


top_pairs = (
    df.sort_values(
        by="similarity_score",
        ascending=False
    )
    .head(10)
)


# Save Top 10
top_pairs.to_csv(
    TOP_10_RESULTS_PATH,
    index=False
)


for rank, (_, row) in enumerate(
    top_pairs.iterrows(),
    start=1
):

    print("\n" + "-" * 80)

    print("Rank:", rank)

    print(
        "Unique ID:",
        row["Unique_ID"]
    )

    print(
        "Similarity Score:",
        round(
            row["similarity_score"],
            4
        )
    )

    print("\nTEXT 1:")
    print(row["text1"])

    print("\nTEXT 2:")
    print(row["text2"])


print("\nTop 10 results saved:")
print(TOP_10_RESULTS_PATH)


# ============================================================
# 21. LEAST SIMILAR TEXT PAIRS
# ============================================================

print("\n" + "=" * 80)
print("13. 5 LEAST SIMILAR TEXT PAIRS")
print("=" * 80)


least_pairs = (
    df.sort_values(
        by="similarity_score",
        ascending=True
    )
    .head(5)
)


least_pairs.to_csv(
    LEAST_5_RESULTS_PATH,
    index=False
)


for rank, (_, row) in enumerate(
    least_pairs.iterrows(),
    start=1
):

    print("\n" + "-" * 80)

    print("Rank:", rank)

    print(
        "Unique ID:",
        row["Unique_ID"]
    )

    print(
        "Similarity Score:",
        round(
            row["similarity_score"],
            4
        )
    )

    print("\nTEXT 1:")
    print(row["text1"])

    print("\nTEXT 2:")
    print(row["text2"])


print("\nLeast similar results saved:")
print(LEAST_5_RESULTS_PATH)


# ============================================================
# 22. SIMILARITY CLASSIFICATION
# ============================================================

print("\n" + "=" * 80)
print("14. SIMILARITY CLASSIFICATION")
print("=" * 80)


def classify_similarity(score):

    if score >= 0.50:
        return "High Similarity"

    elif score >= 0.20:
        return "Moderate Similarity"

    else:
        return "Low Similarity"


df["similarity_category"] = (
    df["similarity_score"]
    .apply(classify_similarity)
)


print("\nSimilarity Category Distribution:")

category_counts = (
    df["similarity_category"]
    .value_counts()
)

print(category_counts)


# ============================================================
# 23. CATEGORY GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

category_counts.plot(
    kind="bar"
)

plt.title(
    "Similarity Category Distribution"
)

plt.xlabel(
    "Similarity Category"
)

plt.ylabel(
    "Number of Text Pairs"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    CATEGORY_GRAPH_PATH
)

plt.show()


print(
    "\nSimilarity category graph saved:"
)

print(
    CATEGORY_GRAPH_PATH
)


# ============================================================
# 24. SAVE FINAL SIMILARITY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("15. SAVING FINAL RESULTS")
print("=" * 80)


df.to_csv(
    SIMILARITY_RESULTS_PATH,
    index=False
)


print(
    "\nSimilarity results saved successfully!"
)

print(
    "Location:",
    SIMILARITY_RESULTS_PATH
)


# ============================================================
# 25. PLAGIARISM DETECTION FUNCTION
# ============================================================

def detect_plagiarism(text1, text2):

    # ----------------------------------------
    # Preprocess both texts
    # ----------------------------------------

    clean_text1 = preprocess_text(text1)

    clean_text2 = preprocess_text(text2)


    # ----------------------------------------
    # Convert texts into TF-IDF vectors
    # ----------------------------------------

    vectors = (
        tfidf_vectorizer.transform(
            [
                clean_text1,
                clean_text2
            ]
        )
    )


    # ----------------------------------------
    # Calculate cosine similarity
    # ----------------------------------------

    score = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]


    # ----------------------------------------
    # Classify similarity
    # ----------------------------------------

    if score >= 0.50:

        category = "High Similarity"

        plagiarism = "Possible Plagiarism"

    elif score >= 0.20:

        category = "Moderate Similarity"

        plagiarism = "Needs Review"

    else:

        category = "Low Similarity"

        plagiarism = "Likely Original"


    return (
        score,
        category,
        plagiarism
    )


# ============================================================
# 26. TEST PLAGIARISM DETECTOR
# ============================================================

print("\n" + "=" * 80)
print("16. TESTING PLAGIARISM DETECTOR")
print("=" * 80)


text1 = """
Machine learning is a branch of artificial intelligence
that allows computers to learn from data.
"""


text2 = """
Machine learning is a field of artificial intelligence
where computers learn from available data.
"""


score, category, plagiarism = (
    detect_plagiarism(
        text1,
        text2
    )
)


print("\nPLAGIARISM DETECTION RESULT")
print("-" * 60)

print(
    "Similarity Score:",
    round(score, 4)
)

print(
    "Similarity Category:",
    category
)

print(
    "Result:",
    plagiarism
)


# ============================================================
# 27. PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PROJECT SUMMARY")
print("=" * 80)


print(
    "\nTotal text pairs analyzed:",
    len(df)
)


print(
    "Vocabulary size:",
    len(feature_names)
)


print(
    "Average similarity score:",
    round(
        df["similarity_score"].mean(),
        4
    )
)


print(
    "Maximum similarity score:",
    round(
        df["similarity_score"].max(),
        4
    )
)


print(
    "Minimum similarity score:",
    round(
        df["similarity_score"].min(),
        4
    )
)


print("\nSimilarity categories:")

print(category_counts)


print("\n" + "=" * 80)
print("NLP PLAGIARISM DETECTION ANALYSIS COMPLETED")
print("=" * 80)




# -----------------------------------------
# 16. USER INPUT PLAGIARISM CHECKER
# -----------------------------------------

print("\n" + "=" * 80)
print("USER INPUT PLAGIARISM CHECKER")
print("=" * 80)

def detect_plagiarism(text1, text2):

    # Preprocess both texts
    clean1 = preprocess_text(text1)
    clean2 = preprocess_text(text2)

    # Convert both texts into TF-IDF vectors
    user_vectors = tfidf_vectorizer.transform(
        [clean1, clean2]
    )

    # Calculate cosine similarity
    score = cosine_similarity(
        user_vectors[0:1],
        user_vectors[1:2]
    )[0][0]

    # Classify similarity
    if score >= 0.50:
        category = "High Similarity"
        plagiarism = "Potential Plagiarism"

    elif score >= 0.20:
        category = "Moderate Similarity"
        plagiarism = "Needs Further Review"

    else:
        category = "Low Similarity"
        plagiarism = "Likely Original"

    return score, category, plagiarism


# Take input from the user

print("\nEnter the first text.")
print("Press ENTER twice when finished:")

text1_lines = []

while True:
    line = input()

    if line == "":
        break

    text1_lines.append(line)

text1 = " ".join(text1_lines)


print("\nEnter the second text.")
print("Press ENTER twice when finished:")

text2_lines = []

while True:
    line = input()

    if line == "":
        break

    text2_lines.append(line)

text2 = " ".join(text2_lines)


# Check plagiarism

score, category, plagiarism = detect_plagiarism(
    text1,
    text2
)


# Display result

print("\n" + "=" * 80)
print("PLAGIARISM DETECTION RESULT")
print("=" * 80)

print("\nSimilarity Score:", round(score, 4))

print("Similarity Percentage:",
      round(score * 100, 2), "%")

print("Similarity Category:", category)

print("Final Result:", plagiarism)

print("=" * 80)

# -----------------------------------------
# GRAPH 3: WORD COUNT BEFORE AND AFTER
# PREPROCESSING
# -----------------------------------------

before_after = pd.DataFrame({
    "Text 1": [
        df["text1_word_count"].mean(),
        df["clean_text1_word_count"].mean()
    ],
    "Text 2": [
        df["text2_word_count"].mean(),
        df["clean_text2_word_count"].mean()
    ]
}, index=[
    "Before Preprocessing",
    "After Preprocessing"
])

before_after.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Average Word Count Before and After Preprocessing")
plt.xlabel("Processing Stage")
plt.ylabel("Average Number of Words")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "results/word_count_before_after.png",
    dpi=300
)

plt.show()


# -----------------------------------------
# GRAPH 4: TOP 10 MOST SIMILAR TEXT PAIRS
# -----------------------------------------

top_10_scores = top_pairs.sort_values(
    by="similarity_score",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_10_scores["Unique_ID"].astype(str),
    top_10_scores["similarity_score"]
)

plt.title("Top 10 Most Similar Text Pairs")
plt.xlabel("Cosine Similarity Score")
plt.ylabel("Unique ID")

plt.xlim(0, 1)

plt.tight_layout()

plt.savefig(
    "results/top_10_similarity_pairs.png",
    dpi=300
)

plt.show()

# -----------------------------------------
# GRAPH 5: TOP 10 TF-IDF WORDS
# -----------------------------------------

top_words = tfidf_scores.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_words["word"][::-1],
    top_words["tfidf_score"][::-1]
)

plt.title("Top 10 TF-IDF Words in the First Document")
plt.xlabel("TF-IDF Score")
plt.ylabel("Word")

plt.tight_layout()

plt.savefig(
    "results/top_tfidf_words.png",
    dpi=300
)

plt.show()

