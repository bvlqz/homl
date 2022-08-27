# Hands-On Machine Learning Notes

- [About](#about)
- [Machine Learning Systems](#machine-learning-systems)
  * [Trained with or without human supervision](#trained-with-or-without-human-supervision)
  * [Learning From a Stream of Incoming Data](#learning-from-a-stream-of-incoming-data)
  * [Reaction to Previously Unseen Data (Instance-Based / Model-Based)](#reaction-to-previously-unseen-data-instance-based--model-based)
  * [Challenges](#challenges)
  * [Testing and Validation](#testing-and-validation)
- [Appendices](#appendices)
  * [Appendix A. Glossary](#appendix-a-glossary)
  * [Appendix B. Acronyms](#appendix-b-acronyms)
  * [Appendix C. Data Repositories](#appendix-b-acronyms)

# About

Notes from the book Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow , 2nd Edition, by Aurélien Géron (O’Reilly). Copyright 2019 Kiwisoft S.A.S., 978-1-492-03264-9.

# Machine Learning Systems
> Programming computers so they can _learn_ from data.

Machine Learning is about making machines get better at some task by learning from data, instead of having to explicitly code rules.

Machine Learning shines when:
* The existing solutions require a lot of fine tunning or long lists of rules.
* Using a traditional approach yields no good solution.
* In a Fluctuating environment.
* You want to get insights about complex problems and large amounts of data (data mining).

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
* Anomaly detection and novelty detection
    * One-class SVM
    * Isolation Forest

* Visualization and dimensionality reduction
  * Principal Component Analysis (PCA)
  * Kernel PCA
  * Locally Linear Embedding (LLE)
  * t-Distributed Stochastic Neighbor Embedding (t-SNE)

* Association rule learning: Dig into large amounts of data and discover interesting relations between attributes
  * Apriori
  * Eclat

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

Learns the training data by heart; then, when given a new instances, it uses a similarity measure to find the most similar learned instances and uses them to make predictions.

### Model-based learning
Build a model with examples and use it to make predictions. Detects patterns in the training data and build a predictive model.

It searches for an optimal value for the model parameters such that the model will generalize well to new instances.

## Challenges
A system will not perform well if the training set is not representative. 
A model needs to be neither too simple (underfit) nor too complex (overfit)

### Insufficient Quantity of Training Data
It takes _a lot_ of data for most Machine Learning algorithms to work properly

### Nonrepresentative Training Data - Sampling Bias
When some members of the population are systematically more likely to be selected in a sample than others, the model will be unlikely to make accurate predictions. 

### Poor Quality Data
If the data is full of errors, outliers and noise, it will make it harder for the system to detect underlying patterns.

### Irrelevant Features
A critical part of the success of a Machine Learning project is coming up with a good set of features to train on. This process is called _feature engineering_:
1. Feature Selection: Select the most useful features to train on among existing features.
2. Feature extraction: Combining features to produce a more useful one, dimensionality reduction algorithms can help.

### Overfitting
The model is too complex relative to the amount and noisiness of the training data. If the training error is low (the model makes few mistakes on the training set) but the generalization error is high, it means that the model is overfitting the training data. 

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


## Testing and Validation

### The Difference Between a Model Parameter and a Learning Algorithm Hyperparameter.
A model has one or more model parameters that determine what it will predict given a new instance (e.g., the slope of a linear model). A learning algorithm tries to find optimal values for these parameters such that the model generalizes well to new instances.

By convention, the Greek letter $\theta$ (theta) is frequently used to represent model parameters.

A hyperparameter is a parameter of the learning algorithm itself, not the model (e.g., the amount of regularization to apply).

### Training Set
Split your data into two sets: the _training set_ and the _test set_.

A labeled training set is a training set that contains the desired solution for each instance.

### _Holdout Validation_
1. Hold out part of the training set to evaluate several candidate models and select the best one. The new held-out set is called the _validation set_.
2. Train multiple models with various hyperparameters on the reduced training set (full training set minus the validation set).
3. Select the model that performs best on the validation set.
4. Train the best model on the full training set (including the validation set).
5. Evaluate this final model on the test set to get an estimate of the generalization error. 

* If the validation set is too small, the model evaluations will be imprecise.
* Is the validation set is too large, the remaining training set will be much smaller than the full training set. One way to solve this problem is to use _cross-validation_, using many small validations sets.

# Appendices

## Appendix A. Glossary

* **Anomaly detection**: Train with mostly normal instances and when it sees a new one it can tell wether it looks like a normal one or an anomaly
* **Attribute**: Describes a data type, attribute and feature are ofter used interchangeably
* **Cost function**: Measure how bad the model is
* **Feature**: Generally means an attribute plus its value
* **Generalization**: How many number of training examples are needed for the system to make good predictions on examples it has never seen before, in order for a model to generalize well, it is crucial that the training data be representative of the new cases that you want the model to generalize to.
* **Holdout validation**: [Jump to section](#holdout-validation)
* **Hyperparamenter**: The amount of regularization to apply during learning (this is a parameter of the learning algorithm, not the model), it is set before training and remains constant during training
* **Hypotesis**: The system's prediction function
* **Learning rate**: how fast the system adapt to changing data, faster learning rate typically means the system will also forget the old data
* **Min-max scaling**: aka. Normalization - Values are shifted and rescaled so that they end up ranging from 0 to 1.
* **Novelty detection**: Detect new instances that look different from all instances in the training set
* **Out-of-core learning**: Train systems on datasets that cannot fit in one machine main memory - load part of the data - run training step on that data - repeat until it has run on all data. 
* **Out-of-sample error**: Generalization error, tells how well the model will perform on instances it has never seen before. If training error is low, but generalization error is high, the model is over-lifting the training data 
* **Overfitting**: [Jump to section](#overfitting)
* **Pipeline**: A sequence of data processing components
* **Predictor**: A feature. If given a task to predict a target numeric value the set of features are called predictors.
* **Regularization**: Constraining a model to make it simpler and reduce the risk of [overfitting](#overfitting).
* **Sampling bias**: [Jump to section](#nonrepresentative-training-data---sampling-bias)
* **Signal**: A piece of information fed to a Machine Learning Algorithm
* **Standard Deviation**: The square root of the variance, which is the average of the squared deviation from the mean
* **Standarization**: Subtract the mean value, and then divide it by the standard deviation so that the resulting distribution has unit variance.
* **Underfitting**: [Jump to section](#underfitting)
* **Utility function (fitness function)**: Measure hoy good the model is
* **Validation Set**: [Jump to section](#holdout-validation)

## Appendix B. Acronyms
* **CNN**: Convolutional Neural Network
* **DBN**: Deep Belief Networks
* **GD**:  Gradient Descent
* **HCA**: Hierarchical Cluster Analysis
* **IID**: Independent and Identically Distributed
* **LASSO Regression**: Least Absolute Shrinkage and Selection Operator Regression
* **LLE**: Locally Linear Embedding
* **MAE**: [**Mean Absolute Error**](#mean-absolute-error) (Also known as Average Absulte Deviation)
* **NLP**: Natural Language Processing
* **NLU**: Natural Language Understanding
* **PCA**: Principal Component Analysis
* **RBM**: Restricted Boltzmann Machines
* **RL**: Reinforcement Learning
* **RMSE**: [**Root Mean Square Error**](#root-mean-square-error)
* **RNN**: Recurrent Neural Network
* **SGD**: Stochastic Gradient Descent
* **SVD**: Singular Value Decomposition
* **SVM**: Support Vector Machines
* **t-SNE**: t-Distributed Stochastic Neighbor Embedding

## Appendix C. Data Repositories

* [UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php)
* [Kaggle Datasets](https://www.kaggle.com/datasets)
* [Registry of Open Data on AWS](https://registry.opendata.aws/)
* [Data Portals: A Comprehensive List of Open Data Portals from Around the World](http://dataportals.org/)
* [OpenDataMonitor](https://opendatamonitor.eu/frontend/web/index.php?r=dashboard%2Findex)
* [Nasdaq Data Link: A premier source for financial, economic and alternative datasets](https://data.nasdaq.com/)
* [List of datasets for machine-learning research](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research)
* [Quora: Where can I find large datasets open to the public?](https://www.quora.com/Where-can-I-find-large-datasets-open-to-the-public)
