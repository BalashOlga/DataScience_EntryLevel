# data_loader.py
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# загрузка данных
def load_dataF_csv(file_name, target_column, test_size = 0.2, random_state = 42 ):
    """
    Загрузка данных из CSV файла.
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


def load_dataF_json(file_name, target_column, test_size = 0.2, random_state = 42 ):
    """
    Загрузка данных из json файла.
    :return: dataFrame с загруженными данными.
    """
    try:    
           dataF = pd.read_json(file_name)
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


def load_dataF_excel(file_name, target_column, test_size = 0.2, random_state = 42 ):
    """
    Загрузка данных из excel файла.
    :return: dataFrame с загруженными данными.
    """
    try:    
           dataF = pd.read_json(file_name)
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

def load_csv_into_bd(connection, table_name, csv_file_path):
    try:
        cursor = connection.cursor()
        with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
            # Читаем CSV-файл
            csv_reader = csv.reader(csv_file)
            
            # Пропускаем заголовок (если он есть)
            header = next(csv_reader, None)
            
            # Формируем SQL-запрос для вставки данных
            placeholders = ', '.join(['%s'] * len(header))  # Плейсхолдеры для параметров
            query = f"INSERT INTO {table_name} ({', '.join(header)}) VALUES ({placeholders})"
            
            # Вставляем данные построчно
            for row in csv_reader:
                cursor.execute(query, row)
                
        # Фиксируем изменения
        connection.commit()
        print(f"Данные из {csv_file_path} успешно загружены в таблицу {table_name}")

    except Exception as e:
        # В случае ошибки откатываем изменения
        connection.rollback()
        print(f"Ошибка загрузки csv-файла: {e}")
    finally:
        # Закрываем курсор
        cursor.close()