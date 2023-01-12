"""
main code that you will run
"""

from linear_model import LogisticRegression
from ensemble import BaggingClassifier
from data_handler import load_dataset, split_dataset
from metrics import accuracy, precision_score, recall_score, f1_score
import argparse

parser = argparse.ArgumentParser(description='Script to run logistic regression with bagging')

parser.add_argument('--data_file', type=str, default="data_banknote_authentication.csv", help="data file name")    
parser.add_argument('--lr', type=float, default=0.05, help="learning rate")
parser.add_argument('--num_iter', type=int, default=1000, help="number of iterations")
parser.add_argument('--verbose', type=bool, default=False, help="verbose")

if __name__ == '__main__':
    
    args = parser.parse_args()
    filename = args.data_file 
    lr = args.lr 
    num_iter = args.num_iter 
    verbose = args.verbose 
    
    # data load
    X, y = load_dataset(filepath="./"+filename)

    # split train and test
    X_train, y_train, X_test, y_test = split_dataset(X, y)

    # training
    params = dict({
        "learning_rate": lr,
        "num_iter": num_iter,
        "verbose": verbose
    })
    print("Parameters: ", params)

    base_estimator = LogisticRegression(params)
    classifier = BaggingClassifier(base_estimator=base_estimator, n_estimator=9)
    classifier.fit(X_train, y_train)

    # testing
    y_pred = classifier.predict(X_test)

    # performance on test set
    print('Accuracy ', accuracy(y_true=y_test, y_pred=y_pred))
    print('Recall score ', recall_score(y_true=y_test, y_pred=y_pred))
    print('Precision score ', precision_score(y_true=y_test, y_pred=y_pred))
    print('F1 score ', f1_score(y_true=y_test, y_pred=y_pred))
