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
from sklearn.metrics import mean_squared_error
import logging
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Удаление дубликатов
def drop_double (dataF):
    df_unique = dataF.drop_duplicates()
    return(df_unique)

# Вывод всех дублирующихся строк
def double (dataF):
    logger.info("Вывод информации о дублях.")
    # Поиск дубликатов и подсчёт их количества
    duplicate_counts = dataF[dataF.duplicated(keep=False)].groupby(dataF.columns.tolist()).size().reset_index(name='count')

    total_duplicates = duplicate_counts['count'].sum() - duplicate_counts.shape[0]
    print("Общее количество дубликатов:", total_duplicates)

    # Подсчёт количества дубликатов
    # value_counts = dataF.value_counts().reset_index(name='count')
       
    # Фильтрация строк, которые встречаются более одного раза (дубликаты)
    # duplicate_counts = value_counts[value_counts['count'] > 1]

    #print("Дубликаты:")
    #print(duplicate_counts)
     
    # duplicate_rows = dataF.loc[dataF.duplicated(keep=False)]
    # №print(duplicate_rows)

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


def get_outliers_iqr(dataF, target_column=None, iqr_coef=1.5, verbose=True):

    # Проверка входных данных
    if not isinstance(dataF, pd.DataFrame) or dataF.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Обработка target_column
    if target_column is None:
        columns_to_check = dataF.columns
    elif isinstance(target_column, str):
        columns_to_check = dataF.columns.drop(target_column, errors='ignore')
    else:
        columns_to_check = dataF.columns.drop(target_column, errors='ignore')
    
    if len(columns_to_check) == 0:
        return pd.DataFrame(), dataF.copy()
    
    # Создаем полную маску выбросов
    outlier_mask = pd.Series(False, index=dataF.index)
    outlier_report = {}
    
    for col in columns_to_check:
        try:
            # Преобразуем в числовой тип, если возможно
            col_data = pd.to_numeric(dataF[col], errors='coerce').dropna()
            
            if len(col_data) < 4:
                continue
                
            # Вычисляем квантили с методом Хая
            Q1 = col_data.quantile(0.25, interpolation='midpoint')
            Q3 = col_data.quantile(0.75, interpolation='midpoint')
            IQR = Q3 - Q1
            
            # Вычисляем границы
            lower_bound = Q1 - iqr_coef * IQR
            upper_bound = Q3 + iqr_coef * IQR
            
            # Создаем маску для текущего столбца
            current_mask = (dataF[col] < lower_bound) | (dataF[col] > upper_bound)
            outlier_mask = outlier_mask | current_mask
            
            # Сохраняем статистику
            outlier_report[col] = {
                'outliers': current_mask.sum(),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min': col_data.min(),
                'max': col_data.max()
            }
        except Exception as e:
            if verbose:
                print(f"Ошибка обработки столбца {col}: {str(e)}")
            continue
    
    # Разделяем данные
    outliers_df = dataF[outlier_mask].copy()
    clean_df = dataF[~outlier_mask].copy()
    
    # Добавляем информацию о выбросах
    if verbose:
        print("\nДетальный отчет о выбросах:")
        for col, stats in outlier_report.items():
            print(f"\nСтолбец: {col}")
            print(f"Границы: [{stats['lower_bound']:.2f}, {stats['upper_bound']:.2f}]")
            print(f"Диапазон данных: [{stats['min']:.2f}, {stats['max']:.2f}]")
            print(f"Найдено выбросов: {stats['outliers']}")
        
        #print(f"\nИтого:")
        #print(f"Всего строк с выбросами: {len(outliers_df)}")
        #print(f"Чистых строк: {len(clean_df)}")
        #print(f"Процент выбросов: {len(outliers_df)/len(dataF)*100:.2f}%")
    
    return outliers_df, clean_df
  
def detect_outliers_isolation(dataF, target_column=None, contamination=0.05):
    """
    Обнаружение выбросов с помощью Isolation Forest
    
    """
    # Исключаем целевой столбец
    if target_column:
        cols = dataF.columns.drop(target_column, errors='ignore')
    else:
        cols = dataF.columns
        
    # Выбираем только числовые столбцы
    numeric_cols = dataF[cols].select_dtypes(include=['number']).columns
        
    if len(numeric_cols) == 0:
        return pd.DataFrame(), dataF.copy()
    
    # Обучаем модель
    clf = IsolationForest(contamination=contamination, random_state=42)
    outliers_mask = clf.fit_predict(dataF[numeric_cols]) == -1
    
    return dataF[outliers_mask], dataF[~outliers_mask]

def get_outliers_combined(dataF, target_column=None, iqr_coef=1.5, iso_contamination=0.05):
    """
    Гибридный метод: сначала IQR, затем Isolation Forest
    """
    # Шаг 1: IQR фильтрация
    outliers_iqr, clean_iqr = get_outliers_iqr(dataF, target_column, iqr_coef, verbose=False)
    
    # Шаг 2: Дополнительная очистка через Isolation Forest
    outliers_iso, clean_final = detect_outliers_isolation(clean_iqr, target_column, iso_contamination)
    
    # Объединяем все выбросы
    total_outliers = pd.concat([outliers_iqr, outliers_iso])
    
    return total_outliers, clean_final

def detect_outliers_three_stages(dataF, target_column=None, params=None, dataF_O=None):
  
    # Установка параметров по умолчанию
    default_params = {
        'iqr_multiplier': 2.5,
        'iso_contamination': 0.1,
        'lof_contamination': 0.1,
        'lof_neighbors': 20
    }
    
    if params:
        default_params.update(params)
    params = default_params

    # Шаг 0: Подготовка данных
    if target_column:
        features = dataF.drop(columns=[target_column])
    else:
        features = dataF.copy()
    
    numeric_cols = features.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        return pd.DataFrame(), dataF.copy()
    
    # Масштабирование
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(features[numeric_cols])
    
    # Шаг 1: IQR с параметром
    q1 = np.percentile(X_scaled, 25, axis=0)
    q3 = np.percentile(X_scaled, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - params['iqr_multiplier'] * iqr
    upper = q3 + params['iqr_multiplier'] * iqr
    iqr_mask = ((X_scaled < lower) | (X_scaled > upper)).any(axis=1)
    
    # Шаг 2: Isolation Forest с параметром
    iso_forest = IsolationForest(
        contamination=params['iso_contamination'],
        random_state=42
    )
    iso_mask = iso_forest.fit_predict(X_scaled) == -1
    
    # Шаг 3: Local Outlier Factor с параметрами
    lof = LocalOutlierFactor(
        n_neighbors=params['lof_neighbors'],
        contamination=params['lof_contamination']
    )
    lof_mask = lof.fit_predict(X_scaled) == -1
    
    # Комбинированная маска
    combined_mask = iqr_mask | iso_mask | lof_mask
    
    return dataF[combined_mask], dataF[~combined_mask]

def analyze_remaining_outliers(original_df, cleaned_df):
    """Анализ оставшихся выбросов"""
    numeric_cols = original_df.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        orig_stats = original_df[col].describe()
        clean_stats = cleaned_df[col].describe()
        
        print(f"\nАнализ столбца: {col}")
        print(f"Исходные границы: [{orig_stats['min']:.2f}, {orig_stats['max']:.2f}]")
        print(f"Новые границы: [{clean_stats['min']:.2f}, {clean_stats['max']:.2f}]")
        print(f"Удалено выбросов: {len(original_df) - len(cleaned_df)}")
        
        # Визуализация
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.boxplot(original_df[col].dropna())
        plt.title(f'Original {col}')
        
        plt.subplot(1, 2, 2)
        plt.boxplot(cleaned_df[col].dropna())
        plt.title(f'Cleaned {col}')
        plt.show()

# если выбросы остаются, возможно это не выбросы а закономерности
def advanced_outlier_detection(df):
    """Использует OneClassSVM для сложных распределений"""
    scaler = RobustScaler()
    X = scaler.fit_transform(df.select_dtypes(include=['number']))
    
    model = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
    preds = model.fit_predict(X)
    
    return df[preds == -1], df[preds == 1]   

def clean_statistic(dataF1, dataF_clean, dataF_outliers):
    """
    Выводит статистику очистки данных с расчетом процента выбросов
    
    Параметры:
    dataF1 - исходный датафрейм
    dataF_clean - очищенный датафрейм
    dataF_outliers - датафрейм с выбросами
    """
    total_rows = len(dataF1)
    clean_rows = len(dataF_clean)
    outlier_rows = len(dataF_outliers)
    
    # Расчет процентов
    outlier_percent = (outlier_rows / total_rows) * 100
    clean_percent = 100 - outlier_percent
    
    # Форматированный вывод
    print("\n" + "="*50)
    print("{:^50}".format("СТАТИСТИКА ОЧИСТКИ ДАННЫХ"))
    print("="*50)
    print(f"{'Исходный размер датафрейма:':<30} {total_rows}")
    print(f"{'Удалено строк (выбросы):':<30} {outlier_rows} ({outlier_percent:.2f}%)")
    print(f"{'Осталось строк:':<30} {clean_rows} ({clean_percent:.2f}%)")
    print("="*50 + "\n")
