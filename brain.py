#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yanickdupuisbinette
"""


import pandas as pd
import numpy as np

from joblib import dump

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split



class Brain(object):

    def __init__(self, dropout=0.1, cv=5, dataset='',parameters=None, layer_and_units=None,
                 delete_data=None, onehot_encoder=None, standard_scaler=None):

        self.cv = cv
        self.dropout = dropout
        self.onehot_encoder = onehot_encoder
        self.standard_scaler = standard_scaler
        self.dataset = pd.read_csv(dataset)
        self.parameters = parameters
        self.delete_data = delete_data
        self.layer_and_units = layer_and_units


    def format_data(self):


        self.X = self.dataset.iloc[:, :-1]
        self.y = self.dataset.iloc[:, -1].values

        preprocess = make_column_transformer(
            (OneHotEncoder(), self.onehot_encoder),
            (StandardScaler(), self.standard_scaler))

        self.X = preprocess.fit_transform(self.X)

        dump(preprocess, 'preprocess.pkl', compress=1)

        self.X = np.delete(self.X, self.delete_data, 1)

        return train_test_split(self.X, self.y , test_size=0.2, random_state=0)


    def build_classifier(self, optimizer):

        classifier = Sequential()


        classifier.add(Dense(units = self.layer_and_units[0], kernel_initializer = 'uniform', activation = 'relu', input_dim = self.X.shape[1]))
        classifier.add(Dropout(rate=self.dropout))

        for i in range(1, len(self.layer_and_units)):
            classifier.add(Dense(units = self.layer_and_units[i], kernel_initializer = 'uniform', activation = 'relu'))
            classifier.add(Dropout(rate=self.dropout))

        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

        return classifier


    def search(self):

        X_train, X_test, y_train, y_test = self.format_data()

        classifier = KerasClassifier(build_fn = self.build_classifier)

        grid_search = GridSearchCV(estimator = classifier,
                                    param_grid = self.parameters,
                                    scoring = 'accuracy',
                                    cv = self.cv,
                                    n_jobs = -1)


        grid_search = grid_search.fit(X_train, y_train)

        best_parameters = grid_search.best_params_
        best_accuracy = grid_search.best_score_

        grid_search.best_estimator_.model.save('best_model.h5')

        y_pred = grid_search.best_estimator_.model.predict(X_test)
        y_pred = (y_pred > 0.5)

        cm = confusion_matrix(y_test, y_pred)


        return best_parameters, best_accuracy, cm







