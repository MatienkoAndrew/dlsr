python3 venv env && source env/vin/activate

> pip3 install pandas numpy matplotlib

> pip3 install sklearn

> pip3 install jupyter

## Tasks 1 (return statistical data about dataset in format 'DataFrame')

> python3 describe.py datasets/dataset_train.csv


> python3 describe.py datasets/dataset_test.csv

## Task 2 (Visualization)

### Task 2.1. (Histogram) 

(Make a script called histogram.[extension] which displays a histogram answering the
next question :
Which Hogwarts course has a homogeneous score distribution between all four houses?)

> python3 histogram.py

### Task 2.2. (Scatter plot)

Make a script called scatter_plot.[extension] which displays a scatter plot answering
the next question :
What are the two features that are similar ?

> python3 scatter_plot.py

### Task 2.3. (Pairplot)

Make a script called pair_plot.[extension] which displays a pair plot or scatter plot
matrix (according to the library that you are using).
From this visualization, what features are you going to use for your logistic regression?

> python3 pair_plot.py

## Task 3 (Logistic Regression)

First one will train your models, it’s called logreg_train.[extension]. It takes
as a parameter dataset_train.csv. . For the mandatory part, you must use the
technique of gradient descent to minimize the error. The program generates a file
containing the weights that will be used for the prediction.

> python3 logreg_train.py datasets/dataset_train.csv

A second has to be named logreg_predict.[extension]. It takes as a parameter
dataset_test.csv and a file containing the weights trained by previous program.

In order to evaluate the performance of your classifier this second program will have
to generate a prediction file houses.csv formatted exactly as follows:

> python3 logreg_predict.py datasets/dataset_test.csv weights.csv

## Bonuses

1. Stochastic Logistic Regression

2. Mini Batch Logistic Regression

3. Graphic of loss function of LR model

> python3 logreg_train.py datasets/dataset_test.csv -l

4. Graphic of loss function to compare models

> python3 logreg_train.py datasets/dataset_test.csv -s

5. Accuracy score of models

> python3 logreg_train.py datasets/dataset_test.csv -k


6. Weights of models

> python3 logreg_train.py datasets/dataset_test.csv -w

