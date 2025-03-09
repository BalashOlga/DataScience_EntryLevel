# plot.py
import matplotlib.pyplot as plt
import seaborn as sns
import logging

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