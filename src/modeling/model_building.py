import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import GradientBoostingClassifier
import yaml
import logging

# logging configure

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        model_building_params = params['model_building']
        logger.debug('max features retrieved')
        return model_building_params
    except FileNotFoundError:
        logger.error('File not found')
        raise
    except yaml.YAMLError as e:
        logger.error('yaml error')
        raise
    except Exception as e:
        logger.error('some error occured')
        raise

# fetch the data from data/processed
train_data = pd.read_csv('./data/features/train_bow.csv')

X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values

# Define and train the XGBoost model
model_building_params = load_params(params_path='params.yaml')
clf = GradientBoostingClassifier(n_estimators=model_building_params['n_estimator'], learning_rate=model_building_params['learning_rate'])
clf.fit(X_train, y_train)

# save
pickle.dump(clf, open('model.pkl','wb'))
