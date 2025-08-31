import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from homemade.utils import evaluate
from sklearn.model_selection import train_test_split

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
       x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,stratify=y,random_state=42)

       models = [
              ("Logistic Regression", LogisticRegression()),
              ("Gaussian Naive Bayes", GaussianNB()),
              ("KNN", KNeighborsClassifier(n_neighbors=5)),
       ]

       for name, model in models:
              train_and_evaluate(model, x_train, y_train, x_test, y_test, name)

if __name__ == "__main__":
    main()

# === Logistic Regression ===
# Accuracy             : 0.999137
# Precision (pos=1)    : 0.859375
# Recall/TPR (pos=1)   : 0.578947
# F1 (pos=1)           : 0.691824

# === Gaussian Naive Bayes ===
# Accuracy             : 0.977567
# Precision (pos=1)    : 0.057808
# Recall/TPR (pos=1)   : 0.810526
# F1 (pos=1)           : 0.107919

# === Random Forest ===
# Accuracy             : 0.999507
# Precision (pos=1)    : 0.971831
# Recall/TPR (pos=1)   : 0.726316
# F1 (pos=1)           : 0.831325