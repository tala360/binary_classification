# Binary Classification (Kaggle Competition)

## Project Overview
This project aims to solve a binary classification problem, where memorable (label:1) and non-memorable (label:0) pictures in Brighton are provided as a training dataset. The goal for the classifier is to predict these labels, but for a test dataset in London i.e. _domain adaptation problem_. <br/>

In this classification problem, we observe training data (X_1,Y_1 )…(X_247,Y_247) where X is a 4608-d feature vector; 4096 of them being deep CNNs features taken from the fc7 activation layer of CaffeNet, and the rest 512 being GIST features, and Y is the label {1,0}. <br/>

A variety of classifiers are tested; such as SVM, DT, etc. but they suffered from the Curse of Dimensionality i.e. low accuracy. <br/>

We are provided with the _confidence_ of each instance (which is the probability of the input to fall in a specific class), and the fact that logistic regression analysis is used to investigate the relationship between a binary response and a set of variables, a logistic regression model seems appropriate - with a bagging classifier.

## Paper
[For more information about the project, download my paper here](https://drive.google.com/file/d/1QA14EAkolyg9100FOOQd4-AWrDJfWU6b/view?usp=sharing)

## Important
Note: the code is in 1 file to reduce compilation time. You are free to restructure it and create your own classes/files.

#### [Download the dataset here](https://www.kaggle.com/t/a127c5f83af644c3b2dca12f6d3cc1f2)

##### This project is protected by an MIT License. For more information, please read the [LICENSE](https://github.com/tala360/binary_classification/blob/main/LICENSE) file.
