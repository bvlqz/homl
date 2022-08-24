# Hands-On Machine Learning

- [About](#about)
- [Part I. Machine Learning](#part-i-machine-learning)
  * [Types of Machine Learning](#types-of-machine-learning)
- [Part II. Appendices](#part-ii-appendices)
  * [Appendix A. Glossary](#appendix-a-glossary)
  * [Appendix B. Acronyms](#appendix-b-acronyms)

# About

Notes from the book Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow , 2nd Edition, by Aurélien Géron (O’Reilly). Copyright 2019 Kiwisoft S.A.S., 978-1-492-03264-9.

# Part I. Machine Learning
## What is Machine Learning
Programming computers so they can _learn_ from data.

## Types of Machine Learning
### Supervised

Trained with human supervision, the data you feed includes the desired solutions
* k-Nearest neighbors
* Linear Regression
* Logistic Regression
* Support Vector Machines
* Decision Trees and Random Forest
* Neural Networks (in some cases)

### Unsupervised

Trained without human supervision, the training data is unlabeled. The system tries to learn without a teacher.

* Clustering
    * K-Means
    * DBSCAN
    * Hierarchical Cluster Analysis
* Anomaly detection and novelty detection
    * Anomaly detection: train with mostly normal instances and when it sees a new one it can tell wether it looks like a normal one or an anomaly
    * Novelty detection: Detect new instances that look different from all instances in the training set
    * One-class SVM
    * Isolation Forest
* Visualization and dimensionality reduction
    * Principal Component Analysis
    * Kernel PCA
    * Locally Linear Embedding 
    * t-Distributed Stochastic Neighbor Embedding (t-SNE)

# Part II. Appendices

## Appendix A. Glossary
**Attribute or Feature**: Describes a data type

**Cost function**: Measure how bad the model is

**Generalization**: How many number of training examples are needed for the system to make good predictions on examples it has never seen before, in order for a model to generalize well, it is crucial that the training data be representative of the new cases that you want the model to generalize to.

**Hyperparamenter**: The amount of regularization to apply during learning (this is a parameter of the learning algorithm, not the model), it is set before training and remains constant during training
**Hypotesis**: The system's prediction function

**Learning rate**: how fast the system adapt to changing data, faster learning rate typically means the system will also forget the old data

**Min-max scaling**: aka. Normalization - Values are shifted and rescaled so that they end up ranging from 0 to 1.

**Out-of-core learning**: Train systems on datasets that cannot fit in one machine main memory - load part of the data - run training step on that data - repeat until it has run on all data. 

**Out-of-sample error**: Generalization error, tells how well the model will perform on instances it has never seen before. If training error is low, but generalization error is high, the model is over-lifting the training data 

**Overfitting**: The model performs well on the training data, but it does not generalize well.

**Pipeline**: A sequence of data processing components

**Regularization**: Constraining a model to make it simpler and reduce the risk of overfitting.

**Signal**: A piece of information fed to a Machine Learning Algorithm

**Standard Deviation**: The square root of the variance, which is the average of the squared deviation from the mean

**Standarization**: Substract the mean value, and then divide it by the standard deviation so that the resulting distribution has unit variance.

**Utility function (fitness function)**: Measure hoy good the model is

**Validation Set**: Used to compare models, it makes it possible to select the best model and tune the hyperparamenters


## Appendix B. Acronyms
**CNN** Convolutional Neural Network

**DBN** Deep Belief Networks

**GD**  Gradient Descent

**HCA** Hierarchical Cluster Analysis

**IID** Independent and Identically Distributed

**LASSO Regression** Least Absolute Shrinkage and Selection Operator Regression

**LLE** Locally Linear Embedding

**MAE** Mean Absolute Error (aka average absulte deviation)

**NLP** Natural Language Processing

**NLU** Natural Language Understanding

**PCA** Principal Component Analysis

**RBM** Restricted Boltzmann Machines

**RL** Reinforcement Learging

**RMQE** Root Mean Square Error

**RNN** Recurrent Neural Network

**SGD** Stochastic Gradient Descent

**SVD** Singular Value Decomposition

**SVM** Support Vector Machines

**t-SNE** t-Distributed Stochastic Neighbor Embedding