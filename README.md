# Car Safety Classification Project

## Overview
This project aims to predict car safety using machine learning models. The dataset contains various features such as buying price, maintenance cost, number of doors, number of persons, luggage boot size, and safety rating. Different classification algorithms including Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), and Support Vector Classifier (SVC) are implemented and compared to determine the most effective model for car safety prediction.

## Dataset
The dataset used in this project includes the following columns:
- buying: buying price of the car
- maint: maintenance cost
- doors: number of doors
- persons: number of persons the car can accommodate
- lug_boot: size of the luggage boot
- safety: safety rating (classes: unacc, acc, good, vgood)

The project utilizes two datasets:
- `cars_train.csv`: Training dataset containing the car attributes used for model training.
- `cars_test.csv`: Testing dataset containing unseen car attributes for evaluating model performance.

To load the datasets into your project, use the following code:
```python
import pandas as pd

# Load the training dataset
cars_train = pd.read_csv('cars_train.csv', header=None)

# Load the testing dataset
cars_test = pd.read_csv('cars_test.csv', header=None)
```

## Installation
To run the project locally, follow these steps:
1. Clone the repository: `git clone https://github.com/your-username/car-safety-classification.git`
2. Navigate to the project directory: `cd car-safety-classification`
3. Install the required dependencies: `pip install -r requirements.txt`

## Final Model: Decision Tree Classifier
We have selected the Decision Tree Classifier as our final model for predicting car safety. Here are the details:
- Algorithm: Decision Tree Classifier
- Criterion: Gini impurity
- Random State: 10
- Splitter: Best

We fitted the model on the training data (x_train, y_train) and used it to predict values.

### Export Decision Tree Model as Graphviz File
We exported the Decision Tree model as a Graphviz file using the export_graphviz function from the sklearn.tree module. This allows us to visualize the Decision Tree's structure and decision-making process.

```python
import graphviz
from sklearn import tree

with open("model_DecisionTree.txt", "w") as f:
    f = tree.export_graphviz(model_DecisionTree, feature_names=cars_train.columns[0:-1], out_file=f)
```
## Usage
- `car_safety_classification.ipynb`: Jupyter Notebook containing the project code and analysis.
- `cars_train.csv`: Training dataset containing the car attributes used for model training.
- `cars_test.csv`: Testing dataset containing unseen car attributes for evaluating model performance.
- `model_DecisionTree.txt`: Graphviz file containing the Decision Tree model.
## Model Performance
The performance of different machine learning models is evaluated as follows:

- DecisionTreeClassifier: 97.83%
- KNeighborsClassifier: 93.50%
- SVC: 98.56%
- LogisticRegression: 64.62%
## Conclusion
Based on the evaluation results, the Decision Tree Classifier achieved an accuracy of 97.83% for car safety classification.

## Credits
This project was developed by Viral Furiya.
