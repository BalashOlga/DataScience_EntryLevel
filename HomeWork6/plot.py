# plot.py
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
import matplotlib.ticker as ticker

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Гистограмма частоты повторений каждого значения в колонке
def hist(dataF, column):
    try:
        logger.info(f"Построение гистограммы для колонки: {column}")
        plt.figure(figsize=(5, 3))  # Уменьшаем размер до 5x3 дюймов
        sns.histplot(dataF[column], bins=len(dataF[column].unique()), kde=False, color='blue', alpha=0.6)
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.title("Горизонтальная гистограмма")
        plt.xlabel("Значение")
        plt.ylabel("Частота")
        plt.show()
        logger.info(f"Гистограмма для колонки {column} успешно построена.")
    except Exception as e:
        logger.error(f"Ошибка при построении гистограммы для колонки {column}: {e}")

# Точечный график (scatter plot)
def scatter(dataF, column, num_points=100):
    try:
        logger.info(f"Построение точечного графика для колонки: {column} (первые {num_points} точек)")
        dataF_column = dataF[column]
        plt.figure(figsize=(5, 3))
        plt.scatter(range(num_points), dataF_column[:num_points], color='blue', label=column)
        plt.title(f'{column} (первые {num_points} точек)')
        plt.legend()
        plt.show()
        logger.info(f"Точечный график для колонки {column} успешно построен.")
    except Exception as e:
        logger.error(f"Ошибка при построении точечного графика для колонки {column}: {e}")

# Столбчатая диаграмма (bar plot)
def bar(dataF, X, y):
    try:
        logger.info(f"Построение столбчатой диаграммы для {X} и {y}")
        plot_data_x = dataF[X]
        plot_data_y = dataF[y]
        plt.figure(figsize=(5, 3))
        plt.bar(plot_data_x, plot_data_y)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.title(f'{X} - {y}')
        plt.xlabel(X)
        plt.ylabel(y)
        plt.show()
        logger.info(f"Столбчатая диаграмма для {X} и {y} успешно построена.")
    except Exception as e:
        logger.error(f"Ошибка при построении столбчатой диаграммы для {X} и {y}: {e}")

# Линейный график (line plot)
def plot(dataF, column, num_points=100):
    try:
        logger.info(f"Построение линейного графика для колонки: {column} (первые {num_points} точек)")
        dataF_column = dataF[column]
        plt.figure(figsize=(5, 3))
        plt.plot(dataF_column[:num_points])
        plt.title(f' {column} (первые {num_points} точек)')
        plt.ylabel(column)
        plt.show()
        logger.info(f"Линейный график для колонки {column} успешно построен.")
    except Exception as e:
        logger.error(f"Ошибка при построении линейного графика для колонки {column}: {e}")

# График ядерной оценки плотности (KDE plot)
def kde(dataF, column, analyzed_column):
    try:
        logger.info(f"Построение KDE графика для колонки: {column} с анализом по {analyzed_column}")
        sns.kdeplot(x=column, data=dataF, hue=analyzed_column, common_norm=False)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.title("Kernel Density Function")
        plt.show()
        logger.info(f"KDE график для колонки {column} успешно построен.")
    except Exception as e:
        logger.error(f"Ошибка при построении KDE графика для колонки {column}: {e}")

# Boxplot для поиска выбросов
def box(dataF, features): 
    plt.figure(figsize=(15, 8))
    for i, feature in enumerate(features, 1):
        plt.subplot(4, 4, i)
        sns.boxplot(y=dataF[feature])
        plt.title(feature)
    plt.tight_layout()
    plt.show()        


def compare_boxplots(dataframes_dict, features, figsize=(10, 6), palette='husl', 
                    rotation=45, show_stats=True, show_outliers=True):
    
    # Проверка признаков
    valid_features = [f for f in features 
                     if any(f in df.columns for df in dataframes_dict.values())]
    
    if not valid_features:
        print("Нет валидных признаков для построения графиков")
        return

    for feature in valid_features:
        plt.figure(figsize=figsize)
        
        # Подготовка данных с сохранением порядка
        plot_data = []
        dataset_order = []  # Сохраняем порядок датасетов
        for name, df in dataframes_dict.items():
            if feature in df.columns:
                temp_df = df[[feature]].copy()
                temp_df['dataset'] = name
                plot_data.append(temp_df)
                dataset_order.append(name)  # Запоминаем порядок
        
        combined_df = pd.concat(plot_data)
        
        # Построение boxplot с явным указанием порядка
        ax = sns.boxplot(
            x='dataset',
            y=feature,
            data=combined_df,
            palette=palette,
            width=0.3,
            linewidth=0.7,
            fliersize=3,
            whis=1.5,
            showfliers=show_outliers,
            order=dataset_order  # Явно задаем порядок
        )
        
        # Настройки оформления
        plt.title(f'Сравнение распределения: {feature}', fontsize=14, pad=15)
        plt.xlabel('Датасет', fontsize=12)
        plt.ylabel(feature, fontsize=12)
        plt.xticks(rotation=rotation, fontsize=11)
        plt.yticks(fontsize=11)
        plt.grid(axis='y', linestyle='-', alpha=0.2, linewidth=0.5)
        
        # Аннотации 
        if show_stats:
            stats = combined_df.groupby('dataset')[feature].describe()
            # Получаем позиции boxplot'ов
            positions = range(len(dataset_order))
            for pos, name in zip(positions, dataset_order):
                if name in stats.index:
                    row = stats.loc[name]
                    txt = f"n={row['count']}\nmed={row['50%']:.1f}\nIQR={row['75%']-row['25%']:.1f}"
                    y_pos = row['min'] * 0.98
                    ax.text(
                        pos, y_pos, txt, 
                        ha='center', va='top', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8, pad=2, boxstyle='round')
                    )
        
        plt.tight_layout()
        plt.show()