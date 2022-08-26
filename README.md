# Hands-On Machine Learning

- [About](#about)
- [Part I. Machine Learning](#part-i-machine-learning)
  * [Types of Machine Learning](#types-of-machine-learning)
- [Part II. Appendices](#part-ii-appendices)
  * [Appendix A. Glossary](#appendix-a-glossary)
  * [Appendix B. Acronyms](#appendix-b-acronyms)

# About

Notes from the book Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow , 2nd Edition, by Aurélien Géron (O’Reilly). Copyright 2019 Kiwisoft S.A.S., 978-1-492-03264-9.

# Machine Learning Systems


Machine Learning is about making machines get better at some task by learning from data, instead of having to explicitly code rules.

## Trained With or Without Human Supervision


### Supervised

Trained with human supervision, the data you feed includes the desired solutions, called labels.

Classification is a typical supervised learning task: For example, a spam filter trained with many examples along with their class (spam or ham)

Some of the most important supervised learning algorithms:

* k-Nearest neighbors
* Linear Regression
* Logistic Regression: It can output a value that corresponds to the probability of belonging to a given class
* Support Vector Machines
* Decision Trees and Random Forest
* Neural Networks (in some cases)

### Unsupervised

Trained without human supervision, the training data is unlabeled. The system tries to learn without a teacher.

* Clustering
  * K-Means
  * DBSCAN
  * Hierarchical Cluster Analysis (HCA)
* Anomaly detection [1] and novelty detection [2]
    * One-class SVM
    * Isolation Forest

[1] Anomaly detection: train with mostly normal instances and when it sees a new one it can tell wether it looks like a normal one or an anomaly

[2] Novelty detection: Detect new instances that look different from all instances in the training set)

* Visualization and dimensionality reduction
  * Principal Component Analysis (PCA)
  * Kernel PCA
  * Locally Linear Embedding (LLE)
  * t-Distributed Stochastic Neighbor Embedding (t-SNE)

* Association rule learning [3]
  * Apriori
  * Eclat

[3] Dig into large amounts of data and discover interesting relations between attributes


### Semisupervised

Usually a combination of supervised and unsupervised algorithms

* Deep Belief Networks (DBNs)
* Restricted Boltzmann Machines (RBMs)

### Reinforcement Learning

The learning system (_agent_) observes the environment, selects and perform actions, and gets _rewards_ or _penalties_ in return. The _agent_ must learn by itself what is the best strategy (_policy_) to get the most reward over time. The _policy_ defines what action the _agent_ should choose when it is in a given situation.

## Learning From a Stream of Incoming Data

### Offline Learning - Batch Learning

The system is incapable of learning incrementally. It must be trained using all the available data. 
When it is launched into production, runs without learning anymore, it applies what is has learned (offline learning)

### Online Learning - Incremental Learning
The system is trained incrementally by feeding it data instances sequentially, either individually or in small groups called mini-batches.

## Reaction to Previously Unseen Data (Instance-Based / Model-Based)

### Instance-based learning
_Learning by heart_

Generalizes to new cases by using a similarity measure to compare them to the learned examples. Compares new data points to known data points.


### Model-based learning
Build a model with examples and use it to make predictions. Detects patterns in the training data and build a predictive model.

## Challenges
### Insufficient Quantity of Training Data
It takes _a lot_ of data for most Machine Learning algorithms to work properly
### Nonrepresentative Training Data - Sampling Bias
When some members of the population are systematically more likely to be selected in a sample than others, the model will be unlikely to make accurate predictions. 
### Poor Quality Data

### Irrelevant Features
A critical part of the success of a Machine Learning project is coming up with a good set of features to train on. This process is called _feature engineering_:
1. Feature Selection: Select the most useful features to train on among existing features.
2. Feature extraction: Combining features to produce a more useful one, dimensionality reduction algorithms can help.
### Overfitting
The model is too complex relative to the amount and noisiness of the training data

* Simplify the model by selecting one with fewer parameters
* Reduce the number of attributes in the training data
* Add constraints to the model to make it simpler (regularization)
* Gather more training data
* Reduce the noise in the training data (e.g., fix data errors and remove outliers)

### Underfitting
The model is too simple to learn the underlying structure of the data

* Select a more powerful model
* Feed better features to the learning algorithm ([**feature engineering**](#irrelevant-features))
* Reduce the constrains on the model






# Appendices

## Appendix A. Glossary
**Attribute**: Describes a data type, attribute and feature are ofter used interchangeably

**Cost function**: Measure how bad the model is

**Feature**: Generally means an attribute plus its value

**Generalization**: How many number of training examples are needed for the system to make good predictions on examples it has never seen before, in order for a model to generalize well, it is crucial that the training data be representative of the new cases that you want the model to generalize to.

**Hyperparamenter**: The amount of regularization to apply during learning (this is a parameter of the learning algorithm, not the model), it is set before training and remains constant during training
**Hypotesis**: The system's prediction function

**Learning rate**: how fast the system adapt to changing data, faster learning rate typically means the system will also forget the old data

**Min-max scaling**: aka. Normalization - Values are shifted and rescaled so that they end up ranging from 0 to 1.

**Out-of-core learning**: Train systems on datasets that cannot fit in one machine main memory - load part of the data - run training step on that data - repeat until it has run on all data. 

**Out-of-sample error**: Generalization error, tells how well the model will perform on instances it has never seen before. If training error is low, but generalization error is high, the model is over-lifting the training data 

[**Overfitting**](#overfitting)

**Pipeline**: A sequence of data processing components

**Predictor**: A feature. If given a task to predict a target numeric value the set of features are called predictors.

**Regularization**: Constraining a model to make it simpler and reduce the risk of [**overfitting**](#overfitting).

[**Sampling bias**](#nonrepresentative-training-data---sampling-bias)

**Signal**: A piece of information fed to a Machine Learning Algorithm

**Standard Deviation**: The square root of the variance, which is the average of the squared deviation from the mean

**Standarization**: Subtract the mean value, and then divide it by the standard deviation so that the resulting distribution has unit variance.

[**Underfitting**](#underfitting)

**Utility function (fitness function)**: Measure hoy good the model is

**Validation Set**: Used to compare models, it makes it possible to select the best model and tune the hyperparamenters


## Appendix B. Acronyms
**CNN**: Convolutional Neural Network

**DBN**: Deep Belief Networks

**GD**:  Gradient Descent

**HCA**: Hierarchical Cluster Analysis

**IID**: Independent and Identically Distributed

**LASSO Regression**: Least Absolute Shrinkage and Selection Operator Regression

**LLE**: Locally Linear Embedding

**MAE**: Mean Absolute Error (Also known as Average Absulte Deviation)

**NLP**: Natural Language Processing

**NLU**: Natural Language Understanding

**PCA**: Principal Component Analysis

**RBM**: Restricted Boltzmann Machines

**RL**: Reinforcement Learning

**RMSE**: Root Mean Square Error

**RNN**: Recurrent Neural Network

**SGD**: Stochastic Gradient Descent

**SVD**: Singular Value Decomposition

**SVM**: Support Vector Machines

**t-SNE**: t-Distributed Stochastic Neighbor Embedding