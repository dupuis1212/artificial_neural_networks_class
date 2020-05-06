#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yanickdupuisbinette
"""

import numpy as np
import pandas as pd

from joblib import load

from keras.models import load_model

preprocess = load('preprocess.pkl')
classifier = load_model('best_model.h5')

######  PREDICT

Xnew = pd.DataFrame(data={
        'CreditScore': [600], 
        'Geography': ['France'], 
        'Gender': ['Male'],
        'Age': [40],
        'Tenure': [3],
        'Balance': [60000],
        'NumOfProducts': [2],
        'HasCrCard': [1],
        'IsActiveMember': [1],
        'EstimatedSalary': [50000]})

Xnew = preprocess.transform(Xnew)
#
Xnew = np.delete(Xnew, [0,3], 1)

new_prediction = classifier.predict(Xnew)
#
# new_prediction = (new_prediction > 0.5)
#
# print(new_prediction)