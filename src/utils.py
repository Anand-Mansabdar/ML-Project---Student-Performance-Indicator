import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor

from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    
    with open(file_path, 'wb') as file_obj:
      dill.dump(obj, file_obj)
      
  except Exception as e:
    raise CustomException(e, sys)
  
def eval_model(X_train, y_train, X_test, y_test, models, params, cv=3):
    try:
        report = {}

        for model_name, model in models.items():

            # 🚫 Skip GridSearchCV for CatBoost
            if isinstance(model, CatBoostRegressor):
                model.fit(X_train, y_train, verbose=False)
                y_test_pred = model.predict(X_test)
                test_model_score = r2_score(y_test, y_test_pred)
                report[model_name] = test_model_score
                continue

            # ✅ GridSearchCV for sklearn models
            gs = GridSearchCV(
                model,
                params[model_name],
                cv=cv,
                n_jobs=-1,
                error_score='raise'
            )

            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            y_test_pred = best_model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)