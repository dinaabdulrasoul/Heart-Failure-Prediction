#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import json
import joblib
import pickle
import os
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path("automl")
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = pd.DataFrame.from_dict(data)
        # make prediction
        result = model.predict(data)
        return result.tolist()
    except Exception as ex:
        error = str(ex)
        return error

