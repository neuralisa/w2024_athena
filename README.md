README

This project provides a collection of PyTorch examples for various machine learning tasks, including linear regression, MNIST classification, Dropout, sentiment analysis, and NER with Word2Vec embeddings.

# Overview

| Task                | Description                                                                 | Author | Status | Path                       |
|---------------------|-----------------------------------------------------------------------------|--------|--------|----------------------------|
| Linear Regression  | A basic example showing how to perform linear regression in PyTorch | Bayar | TODO | |
| MNIST Classifier | A simple neural network to classify handwritten digits from the MNIST dataset. | Bayar | TODO | |
| Dropout example | Demonstrates how dropout regularization works to prevent overfitting in neural networks. | Bayar | TODO | |
| [Sentiment Analysis](https://github.com/olokshyn/tuwien-dl-nlp/blob/master/exercises/word2vec_sentiment_classification.ipynb)  | A text classification task that predicts classifies tweets into Positive, Negative, or Neutral. Word2Vec embeddings | Oleh   | Done | `tuwien-dl-nlp/exercises/word2vec_sentiment_classification.ipynb`    |
| [NER Classification](https://github.com/olokshyn/tuwien-dl-nlp/blob/master/exercises/word2vec_ner_classification.ipynb)  | Shows how to apply Word2Vec embeddings for Named Entity Recognition (NER) on CoNLL 2003 dataset.  | Oleh   | Done | `tuwien-dl-nlp/exercises/word2vec_ner_classification.ipynb`          |

# How to run it
1. Clone this repository
2. Initialize submodules: `git submodule update --init --recursive`
3. Go to the `tuwien-dl-nlp` directory and see the [./tuwien-dl-nlp/README.md](https://github.com/olokshyn/tuwien-dl-nlp/blob/master/README.md) on how to install dependencies.

Homework

Your task is to improve the NER neural network by experimenting with different hyperparameters and settings. Try adjusting learning rates, model architectures, or embedding sizes to enhance performance.
