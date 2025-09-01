import pandas as pd
from homemade.logistic_regression import LogisticRegression
from homemade.naive_bayes import GaussianNB
from homemade.kNN import KNeighborsClassifier
from homemade.utils import evaluate, train_test_split

import warnings
warnings.filterwarnings("ignore")

def read_data(file_path):
       card = pd.read_csv(file_path)
       card_d=card.copy()
       card_d.drop_duplicates(subset=None, inplace=True)
       card=card_d
       del card_d
       return card

def train_and_evaluate(model, x_train, y_train, x_test, y_test, name: str):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"\n=== {name} ===")
    evaluate(y_test, y_pred)

def main():
       card = read_data('creditcard.csv')
       new_features = card[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
              'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V20','V21', 'V22', 'V23', 'V25', 'V26', 'V27','Class']]
       x=new_features.iloc[:,:-1]
       y=new_features.iloc[:,-1]
       x_train,x_test,y_train,y_test=train_test_split(X=x, y=y, stratify=y, test_size=0.2, random_state=42, shuffle=True)

       models = [
              ("Logistic Regression", LogisticRegression()),
              ("Gaussian Naive Bayes", GaussianNB()),
              ("KNN", KNeighborsClassifier(k=5)),
       ]

       for name, model in models:
              train_and_evaluate(model, x_train, y_train, x_test, y_test, name)

if __name__ == "__main__":
    main()

# === Logistic Regression ===
# Accuracy             : 0.999084
# Precision (pos=1)    : 0.852459
# Recall/TPR (pos=1)   : 0.547368
# F1 (pos=1)           : 0.666667

# === Gaussian Naive Bayes ===
# Accuracy             : 0.978677
# Precision (pos=1)    : 0.062058
# F1 (pos=1)           : 0.115497

# === KNN ===
# Accuracy             : 0.999630
# Precision (pos=1)    : 0.962500
# Recall/TPR (pos=1)   : 0.810526
# F1 (pos=1)           : 0.880000