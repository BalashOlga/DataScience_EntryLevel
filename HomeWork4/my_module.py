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
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# вывод статистической информации
def statistic_dataF(dataF) -> None:
    logger.info("Вывод статистической информации о данных.")
    
    # Просмотр первых строк данных
    logger.info("Первые строки данных:")
    print(dataF.head(), end="\n\n")

    # Основная информация о данных
    logger.info("Основная информация о данных:")
    print(dataF.info(), end="\n\n")

    # Описательная статистика
    logger.info("Описательная статистика:")
    print(dataF.describe(include='all'), end="\n\n")

    # Проверка на пропущенные значения
    logger.info("Проверка на пропущенные значения:")
    print(dataF.isnull().sum(), end="\n\n")
    

# предобработка данных
def preprocess_dataF(dataF) -> None:
    logger.info("Начало предобработки данных.")
    
    if dataF.X_train is None or dataF.X_test is None:
        logger.error("Данные не загружены. Вызовите метод load_dataF().")
        return

    dataF.X_train = dataF.scaler.fit_transform(dataF.X_train)
    dataF.X_test = dataF.scaler.transform(dataF.X_test)
    logger.info("Данные успешно предобработаны.")


# кодируем категориальные значения числом
def encoder(dataF, args) -> None:
    logger.info(f"Кодирование категориальных признаков: {args}")
    
    for i in args:
        if dataF[i].dtype == 'object':
            label_encoder = LabelEncoder()
            dataF[i] = label_encoder.fit_transform(dataF[i])
            logger.info(f"Признак {i} успешно закодирован.")
    

# вывод уникальных значений колонки
def unique_values(dataF, column):
    logger.info(f"Вывод уникальных значений для колонки {column}.")
    print(dataF[column].unique())


# подсчет количества null в каждом столбце
def count_missing_values(dataF):
    logger.info("Подсчет количества пропущенных значений в каждом столбце.")
    print(dataF.isnull().sum())    
    

# Находим максимальное значение во всём датафрейме
def max_in_column(dataF, column):
    logger.info(f"Поиск максимального значения в колонке {column}.")
    return dataF[column].max()


# замена null на среднее по столбцу
def replace_nan_to_mean(dataF, column):
    logger.info(f"Замена пропущенных значений в колонке {column} на среднее значение.")
    
    mean_value = dataF[column].mean()  # Вычисляем среднее значение столбца
    dataF.fillna({column: mean_value}, inplace=True) # Заменяем NaN на среднее
    logger.info(f"Пропущенные значения в колонке {column} заменены на среднее значение: {mean_value}.")