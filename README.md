# Multi-Dataset Conditional Text Generation and Augmentation for Sentiment & Emotion Classification

This repository implements a unified pipeline across three datasets—TweetEval, GoEmotions, and DAIR Emotion—performing conditional text generation with SmolLM-135 for data augmentation and fine-tuning transformer classifiers to address class imbalance and improve performance on minority classes.

## Project Overview

- **Datasets**  
  - **TweetEval**: Three-way sentiment classification (Negative, Neutral, Positive)  
  - **GoEmotions**: Twenty-seven fine-grained emotion classes  
  - **DAIR Emotion**: Six core emotions (sadness, joy, love, anger, fear, surprise)

- **Preprocessing**  
  - Text cleaning: lowercase, remove URLs, mentions, hashtags, emojis, non-alphanumeric characters  
  - Control tokens prepended (e.g., `[POSITIVE]`, `[JOY]`, `[SADNESS]`)  
  - Tokenization with fixed max length, padding, and attention masks

- **Conditional Generation for Augmentation**  
  - Fine-tune SmolLM-135M-Instruct on each dataset with sentiment/emotion prefixes  
  - Generate synthetic examples for underrepresented classes using diverse prompt stems  
  - Achieve balanced class distributions across all datasets

- **Classifier Training**  
  - RoBERTa-based sequence classification models  
  - Hyperparameter tuning via Optuna (learning rate, epochs, batch size)  
  - Mixed-precision training and early stopping based on validation metrics

- **Evaluation**  
  - Generation models: training and eval loss, perplexity  
  - Classification models: precision, recall, F1-score, accuracy, and confusion matrices  
  - Special focus on per-class improvements for originally minority labels

## Pipeline Steps

1. **Data Loading & Cleaning**  
   - Load train/validation/test splits from Hugging Face  
   - Clean text and map dataset-specific labels to unified control tokens  
   - Visualize initial class distributions  

2. **Conditional Text Generation & Augmentation**  
   - Fine-tune generation models on each dataset  
   - Use diverse prompt stems to generate synthetic samples for minority classes  
   - Combine original and synthetic data to form balanced datasets  

3. **Classifier Fine-Tuning**  
   - Tokenize combined datasets for classification  
   - Train RoBERTa models on original vs. augmented data  
   - Optimize hyperparameters with Optuna  
   - Select best checkpoints via early stopping on validation F1

4. **Evaluation & Reporting**  
   - Compute classification reports and confusion matrices  
   - Compare performance before and after augmentation  
   - Highlight gains in minority-class recall and F1-score  

## Key Results

- **TweetEval**  
  - Original accuracy: ~0.72, macro F1: ~0.71  
  - Augmented accuracy: ~0.79, macro F1: ~0.79  

- **GoEmotions** (single-label subset)  
  - Original accuracy: ~0.58, macro F1: ~0.51  
  - Augmented accuracy: ~0.83, macro F1: ~0.83  

- **DAIR Emotion**  
  - Original accuracy: ~0.93, macro F1: ~0.90  
  - Augmented accuracy: ~0.97, macro F1: ~0.97  

Augmentation consistently improves minority-class performance and overall balance.

## Repository Structure

- **dairemotion/**: Notebook and scripts for DAIR Emotion dataset  
- **goemotion/**: Notebook and scripts for GoEmotions dataset  
- **tweeteval/**: Notebook and scripts for TweetEval dataset  

Each folder contains a complete Jupyter notebook and supporting files for data preparation, generation, augmentation, and classification.

## Contact

For questions or contributions, please open an issue or submit a pull request.
