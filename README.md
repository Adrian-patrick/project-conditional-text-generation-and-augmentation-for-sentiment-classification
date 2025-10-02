based on the images edit this read me file 
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

### On GoEmotions Dataset

- Original distribution showed significant class imbalance with "neutral" and "admiration" dominating.
- Augmentation equalized emotion samples across 26 classes.
- Classification on balanced dataset achieved an accuracy of ~0.28 with a macro F1 score around 0.28 for 27 classes, showing that emotion classification is a challenging multi-label task but balanced data helps model learning.

### On TweetEval Dataset

- Initial class distribution skewed towards negative and neutral classes.
- Augmented data achieved near-uniform distribution for positive, neutral, and negative labels.
- RoBERTa classification on augmented data reached accuracy of 72% with macro F1 score around 0.73.
- Individual class F1 scores improved substantially post augmentation (from ~0.20-0.50 range to above ~0.60 for all classes).

### On DAIR-AI Dataset

- Original dataset imbalance addressed by synthetic sample generation for minority classes: sadness, joy, love, anger, fear, surprise.
- After augmentation, emotional classes were balanced around 5700 samples each.
- Classification accuracy improved to around 90%.
- Per-class precision and recall ranged from 0.79 (anger) to 0.98 (sadness and joy), with macro F1 scores around 0.90, indicating strong balanced performance.

---
- Augmentation markedly improves class balance, reducing bias toward majority labels.
- Multi-label and multi-class classification accuracy significantly improves post augmentation, especially on minority classes.
- Macro-averaged F1 scores for core emotion classes reach values often above 0.80 on curated datasets.
- Precision and recall balance indicate robust performance in realistic, noisy social media text.
