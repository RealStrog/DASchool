
import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

df = pd.read_csv(r'C:\Users\USER\Downloads\ML_LEC3\Streamlit\housing.csv')

if st.header('Выберите размер обучающей выборки'):
    size = st.slider('Размер обучающей выборки?', 0.1, 0.5, 0.2)
    st.write("Размер обучающей выборки", size )
if st.button('Отобразить первые пять строк'):
    st.write(df.head())

if st.button('Обучить модель'):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1),
                                                        df['MEDV'],
                                                        test_size=size,
                                                        random_state=2100)
    st.write('Разделили данные и передали в обучение')
    regr_model = XGBRegressor()
    regr_model.fit(X_train, y_train)
    pred = regr_model.predict(X_test)
    st.write('Обучили модель, MAE = ' + str(mean_absolute_error(y_test, pred)))
