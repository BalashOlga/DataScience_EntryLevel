# data_loader.py
import pandas as pd
import csv
import sys
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка данных из CSV файла
def load_dataF_csv(file_name, target_column, test_size=0.2, random_state=42):
    """
    Загрузка данных из CSV файла.
    :return: dataFrame с загруженными данными.
    """
    try:
        logger.info(f"Загрузка данных из CSV файла: {file_name}")
        dataF = pd.read_csv(file_name)
       # dataF.X_train = dataF.X_test = dataF.y_train = dataF.y_test = None

        # Создаем X (признаки) и y (целевая переменная)
       # X = dataF.drop(target_column, axis=1)  # Все столбцы, кроме target_column
       # y = dataF[target_column]  # Только столбец target_column

       # dataF.X_train, dataF.X_test, dataF.y_train, dataF.y_test = train_test_split(
        #    X, y, test_size=test_size, random_state=random_state
        #)

        logger.info("Данные успешно загружены")
       # logger.info(f"Матрица признаков X:\n{X}")
       # logger.info(f"Целевая переменная y:\n{y}")

    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        sys.exit(1)
    return dataF


# Загрузка данных из JSON файла
def load_dataF_json(file_name, target_column, test_size=0.2, random_state=42):
    """
    Загрузка данных из JSON файла.
    :return: dataFrame с загруженными данными.
    """
    try:
        logger.info(f"Загрузка данных из JSON файла: {file_name}")
        dataF = pd.read_json(file_name)
        dataF.X_train = dataF.X_test = dataF.y_train = dataF.y_test = None

        # Создаем X (признаки) и y (целевая переменная)
        X = dataF.drop(target_column, axis=1)  # Все столбцы, кроме target_column
        y = dataF[target_column]  # Только столбец target_column

        dataF.X_train, dataF.X_test, dataF.y_train, dataF.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info("Данные успешно загружены и разделены.")
        logger.info(f"Матрица признаков X:\n{X}")
        logger.info(f"Целевая переменная y:\n{y}")

    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        sys.exit(1)
    return dataF


# Загрузка данных из Excel файла
def load_dataF_excel(file_name, target_column, test_size=0.2, random_state=42):
    """
    Загрузка данных из Excel файла.
    :return: dataFrame с загруженными данными.
    """
    try:
        logger.info(f"Загрузка данных из Excel файла: {file_name}")
        dataF = pd.read_excel(file_name)
        dataF.X_train = dataF.X_test = dataF.y_train = dataF.y_test = None

        # Создаем X (признаки) и y (целевая переменная)
        X = dataF.drop(target_column, axis=1)  # Все столбцы, кроме target_column
        y = dataF[target_column]  # Только столбец target_column

        dataF.X_train, dataF.X_test, dataF.y_train, dataF.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info("Данные успешно загружены и разделены.")
        logger.info(f"Матрица признаков X:\n{X}")
        logger.info(f"Целевая переменная y:\n{y}")

    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        sys.exit(1)
    return dataF


# Загрузка данных из CSV в базу данных
def load_csv_into_bd(connection, table_name, csv_file_path):
    try:
        logger.info(f"Загрузка данных из CSV файла {csv_file_path} в таблицу {table_name}")
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
        logger.info(f"Данные из {csv_file_path} успешно загружены в таблицу {table_name}")

    except Exception as e:
        # В случае ошибки откатываем изменения
        connection.rollback()
        logger.error(f"Ошибка загрузки CSV-файла: {e}")
    finally:
        # Закрываем курсор
        cursor.close()