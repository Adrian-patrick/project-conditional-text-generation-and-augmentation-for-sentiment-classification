Multi-Dataset Conditional Text Generation and Augmentation for Sentiment Classification
A robust deep learning pipeline for emotion and sentiment classification using advanced data augmentation through conditional text generation. This project harmonizes, augments, and classifies emotions in text from multiple datasets with transformer models, delivering state-of-the-art results for imbalanced NLP tasks.

Overview
This repository integrates GoEmotions, TweetEval, and DAIR-AI datasets for unified multi-emotion classification. Conditional text generation using GPT-2 augments minority emotion classes, and a RoBERTa classifier is fine-tuned for accurate sentiment recognition. The modular notebooks offer reproducible workflows for preprocessing, augmentation, training, and evaluation.

Key Components
goemotionsfinal.ipynb: Processes, augments, and classifies the GoEmotions dataset.

dairemotionfinal.ipynb: Handles the DAIR-AI dataset with harmonization and conditional augmentation.

tweetevalfinal.ipynb: Preprocesses and classifies TweetEval, including baseline and augmented runs.

gans-robertaworking1.ipynb: RoBERTa-based emotion classification (original and augmented data).

gans-distilgpt2working1.ipynb: Conditional GPT-2 text generation for emotion augmentation.

Pipeline Highlights
Data Harmonization: Standardizes and maps emotion labels across datasets for unified analysis.

Preprocessing: Cleans noisy social text (lowercase, remove URLs/usernames/punctuation).

Conditional Generation: Fine-tunes GPT-2 with emotion tokens; generates new samples for minority classes.

Augmentation: Merges synthetic and real data for balanced training sets.

Transformer Classification: Fine-tunes RoBERTa on merged datasets with full evaluation metrics.

Evaluation: Output includes precision, recall, F1, and per-class statistics for both original and augmented experiments.

Getting Started
Environment

bash
conda create -n emotion-nlp python=3.10
conda activate emotion-nlp
pip install torch torchvision torchaudio transformers datasets scikit-learn pandas matplotlib tqdm
Data

Download GoEmotions, TweetEval, DAIR-AI.

Place the data in a data/ folder. Update paths in notebooks as needed.

Execution

Run each notebook sequentially per dataset: preprocessing → augmentation → classification.

Evaluate performance and inspect generated samples and reports directly in the notebook outputs.

Results
Achieves strong gains for minority class recall and macro F1 on all datasets.

Demonstrates the advantage of synthetic conditional data in boosting classifier robustness.
