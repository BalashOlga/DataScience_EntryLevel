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

def hist(dataF, column):
    #plt.figure(figsize=(10, 6), constrained_layout=True)  # constrained_layout=True
    sns.histplot(dataF[column], bins=5, kde=False, color='blue', alpha=0.6)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.title("Горизонтальная гистограмма")
    plt.xlabel("Значение")
    plt.ylabel("Частота")
    plt.show()


# загрузка данных
def load_dataF(file_name, target_column, test_size = 0.2, random_state = 42 ):
    """
    Загрузка данных из CSV файла.
    :param file_path: Путь к CSV файлу.
    :return: dataFrame с загруженными данными.
    """
    try:    
           dataF = pd.read_csv(file_name)
           dataF.model = LinearRegression()
           dataF.scaler = StandardScaler()
           dataF.X_train = dataF.X_test = dataF.y_train = dataF.y_test = None
           # Создаем X (признаки) и y (целевая переменная)
           X = dataF.drop(target_column, axis=1)  # Все столбцы, кроме target_column
           y = dataF[target_column]  # Только столбец target_column
              
           dataF.X_train, dataF.X_test, dataF.y_train, dataF.y_test = train_test_split(
                X, y, test_size = test_size, random_state = random_state
           )
     
           print("Данные успешно загружены и разделены.")
           print("Матрица признаков X:")
           print(X)
     
           print("\nЦелевая переменная y:")
           print(y)

    except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            sys.exit(1)
    return dataF



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