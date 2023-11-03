"""
Content of preprocess_data.py
"""
from sklearn.model_selection import train_test_split
import pandas as pd 

def prepare_data(path_to_data):

    # Read data from path
    data = pd.read_csv(path_to_data)
    print("\n1. First 7 rows: \n")
    print(data.head(7))

    print("\n2. Last 7 rows: \n")
    print(data.tail(7))    

    X = data.drop('Outcome',axis=1)
    y = data['Outcome']

    return {'features':X, 
            'label':y}


def create_train_test_data(X, y, test_size, random_state):

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)

    return {'x_train': X_train, 'x_test': X_test,
            'y_train': y_train, 'y_test': y_test}