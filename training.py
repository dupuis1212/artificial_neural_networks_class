#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yanickdupuisbinette
"""


import brain


parameters = {'batch_size': [32],'epochs': [100],'optimizer': ['adam']}
onehot_encoder = ['Geography', 'Gender']
standard_scaler = ['CreditScore', 'Age', 'Tenure', 'Balance','NumOfProducts', 'HasCrCard', 'IsActiveMember','EstimatedSalary']
delete_data = [0,3]
layer_and_units = [6,8]

new_brain = brain.Brain(parameters=parameters,
              dataset='data/Churn_Modelling.csv',
              delete_data=delete_data,
              layer_and_units=layer_and_units,
              onehot_encoder=onehot_encoder,
              standard_scaler=standard_scaler)

best_parameters, best_accuracy, cm = new_brain.search()


