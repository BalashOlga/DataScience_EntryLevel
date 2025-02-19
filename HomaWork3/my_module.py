# my_module.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

label_encoder = LabelEncoder();

# вывод статистической информации
def statistic_dataF(dataF) -> None:

    # Просмотр первых строк данных
    print(dataF.head(), end="\n\n")

    # Основная информация о данных
    print(dataF.info(), end="\n\n")

    # Описательная статистика
    print(dataF.describe(include='all'), end="\n\n")

    # Проверка на пропущенные значения
    print(dataF.isnull().sum(), end="\n\n")
    

# предобработка данных
def preprocess_dataF(dataF) -> None:
        """
        масштабирование признаков.
        """
        if dataF.X_train is None or dataF.X_test is None:
            print("Данные не загружены. Вызовите метод load_dataF().")
            return

        dataF.X_train = dataF.scaler.fit_transform(dataF.X_train)
        dataF.X_test = dataF.scaler.transform(dataF.X_test)
        print("Данные успешно предобработаны.")


# кодируем категориальные значения числом
def encoder(dataF, args) -> None:
       for i in args:
           dataF[i + ' (Encoded)'] = label_encoder.fit_transform(dataF[i])
    

# вывод уникальных значений колонки
def unique_values(date, column):
    print(dataF[column].unique())


# подсчет количества null в каждом столбце
def count_missing_values(dataF):
    print(dataF.isnull().sum())    
    

# Находим максимальное значение во всём датафрейме
def max_in_column(dataF, column):
    return dataF[column].max()

# замена null на среднее по столбцу
def replace_nan_to_mean (dataF, column):
    mean_value = dataF[column].mean()  # Вычисляем среднее значение столбца
    dataF.fillna({column: mean_value}, inplace=True) # Заменяем NaN на среднее