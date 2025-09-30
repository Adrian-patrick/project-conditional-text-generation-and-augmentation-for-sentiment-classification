# Multi-Dataset Conditional Text Generation and Augmentation for Sentiment Classification

This project presents a comprehensive pipeline that harmonizes multiple emotion and sentiment datasets, augments minority classes via conditional text generation with GPT-2, and fine-tunes transformer-based classifiers to improve multilingual sentiment and emotion analysis.

---

## Project Overview

- **Datasets:** Integrates GoEmotions, TweetEval, and DAIR-AI datasets with custom label mappings for unified emotion classes.
- **Preprocessing:** Performs thorough cleaning, normalization, and tokenization tailored for social media and informal text.
- **Conditional Text Generation:** Fine-tunes GPT-2 / DistilGPT-2 models to generate synthetic samples for underrepresented emotion classes, balancing the datasets.
- **Classifier Training:** Fine-tunes RoBERTa classifiers on original and augmented data, using hyperparameter optimization (Optuna) and mixed precision.
- **Evaluation:** Detailed performance metrics such as precision, recall, F1-score, and accuracy with special focus on improvements for minority classes after augmentation.

---

## Pipeline Steps

### 1. Data Preparation
- Load datasets from HuggingFace or official sources.
- Clean text: lowercase, remove special characters, URLs, emojis, mentions.
- Map dataset-specific labels to a common emotion scheme covering core emotions like joy, sadness, anger, fear, surprise, love, etc.

### 2. Data Exploration and Imbalance Analysis
- Analyze class distribution identifying majority and minority classes in each dataset.
- Target minority classes for augmentation based on relative imbalance.

### 3. Conditional Text Generation for Augmentation
- Prepare training data by prefixing text samples with emotion labels.
- Fine-tune GPT-2 language models on the conditioned dataset with hyperparameter tuning.
- Generate synthetic examples for minority classes to achieve balanced class distributions.
- Augmented datasets show near-equal representation of all emotion labels.

### 4. Classifier Fine-Tuning and Evaluation
- Tokenize text data for RoBERTa classification.
- Train models on both original and augmented datasets.
- Utilize validation sets for early stopping and selection of best checkpoints.
- Evaluate with classification reports, highlighting improvements in F1-score and recall for formerly underrepresented emotions.

---

## Results Summary

- Augmentation markedly improves class balance, reducing bias toward majority labels.
- Multi-label and multi-class classification accuracy significantly improves post augmentation, especially on minority classes.
- Macro-averaged F1 scores for core emotion classes reach values often above 0.80 on curated datasets.
- Precision and recall balance indicate robust performance in realistic, noisy social media text.
