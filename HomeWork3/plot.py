
# plot.py
import matplotlib.pyplot as plt
import seaborn as sns

# гистограмма частоты повторений каждого значения в колонке
def hist(dataF, column):
    #plt.figure(figsize=(10, 6), constrained_layout=True)  # constrained_layout=True
    sns.histplot(dataF[column], bins=5, kde=False, color='blue', alpha=0.6)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.title("Горизонтальная гистограмма")
    plt.xlabel("Значение")
    plt.ylabel("Частота")
    plt.show()

def scatter(dataF, column, num_points=100):
    dataF_column = dataF[column]
    plt.figure(figsize=(5, 3))
    plt.scatter(range(num_points), dataF_column[:num_points], color='blue', label=column)
    plt.title(f'{column} (первые {num_points} точек)')
    plt.legend()
    plt.show()


def bar(dataF, X, y):
    plot_data_x = dataF[X]
    plot_data_y = dataF[y]
    plt.figure(figsize=(5, 3))
    plt.bar(plot_data_x, plot_data_y)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.title(f'{X} - {y}')
    plt.xlabel(X)
    plt.ylabel(y)
    plt.show()


def plot(dataF, column, num_points=100):
    dataF_column = dataF[column]
    plt.figure(figsize=(10, 6))
    plt.plot(dataF_column[:num_points])
    plt.title(f' {column} (первые {num_points} точек)')
    plt.ylabel(column)
    plt.show()  

def kde(dataF, column, analyzed_column):
    sns.kdeplot(x = column, data = dataF, hue = analyzed_column, common_norm = False);
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.title("Kernel Density Function")
    plt.show() 
 