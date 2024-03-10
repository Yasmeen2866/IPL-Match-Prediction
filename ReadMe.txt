README
This repository contains code for developing a machine learning model to predict cricket scores based on various features. Below is a guide to understanding the contents of this repository.

Importing Libraries
This section imports necessary Python libraries for data manipulation, visualization, preprocessing, and model development. Libraries such as Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Keras, and TensorFlow are imported.

Loading Dataset
The dataset used for this project is ipl_data.csv, which contains various features related to cricket matches such as venue, batting team, bowling team, batsman, bowler, and total score. Pandas is used to load the dataset, and its structure and information are displayed.

Preprocessing Data
This section involves preprocessing steps such as dropping certain features deemed unnecessary for model development. Features such as date, runs, wickets, overs, runs_last_5, wickets_last_5, mid, striker, and non-striker are dropped from the dataset.

Label Encoding
Categorical features in the dataset are encoded using LabelEncoder from Scikit-learn to convert them into numerical representations suitable for machine learning algorithms.

Train Test Split Data
The dataset is split into training and testing sets using train_test_split from Scikit-learn.

Feature Scaling
Features in the dataset are scaled using Min-Max scaling to bring them within a similar range, ensuring better performance of the machine learning model.

Define Neural Network Model
A neural network model is defined using the Keras Sequential API. The model consists of input, hidden, and output layers, with ReLU activation functions for hidden layers and linear activation function for the output layer. Huber loss function is used for model compilation.

Model Training
The defined neural network model is trained using the training data. The model's performance is evaluated based on loss metrics, and a plot is generated to visualize the training and validation loss over epochs.

Model Evaluation
The trained model is evaluated using mean absolute error and mean squared error metrics to assess its performance in predicting cricket scores.

Predict Score Widget
An interactive widget is created using ipywidgets to allow users to select cricket match parameters such as venue, batting team, bowling team, batsman, and bowler. The trained model then predicts the score based on these inputs, providing a convenient tool for score prediction.

Dependencies
Python 3.x
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Keras
TensorFlow
ipywidgets

Note
This project is an adaptation of code sourced from another repository. Credit and acknowledgment go to the original source for providing the foundational code.