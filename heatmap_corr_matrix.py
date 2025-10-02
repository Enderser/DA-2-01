import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from DA_1_21 import get_dataset, calculate_correlation, URL, COLUMNS


def draw_heatmap_with_coolwarm(data: pd.DataFrame, save_name: str | None = None) -> None:
    '''
    Функция для построения тепловой карты (heatmap) для заданных данных 
    с возможностью сохранения графика в файл.

    Параметры
    ---------
    data : pandas.DataFrame
        Данные для построения тепловой карты. Ожидается матрица корреляции 
        или другая числовая матрица
    save_name : str | None, optional, по умолчанию = None
        Имя файла для сохранения графика. Если save_name = None или "", 
        сохранение не производится

    Действие
    --------
    - Строит тепловую карту с использованием цветовой схемы 'coolwarm'
    - Добавляет аннотации с числовыми значениями
    - Автоматически подбирает размер графика на основе размера данных
    - Отображает график и при необходимости сохраняет в файл

    Исключения
    ----------
    Выводит сообщение об ошибке при возникновении исключения в процессе построения графика
    '''
    try:
        # Базовые размеры графика
        base_width = 8
        base_height = 6
        scale_factor = 0.3  # Коэффициент масштабирования для адаптации размера
        
        # Расчет оптимального размера графика на основе размера данных
        n_cols, n_rows = data.shape
        fig_width = base_width + (n_cols * scale_factor)
        fig_height = base_height + (n_rows * scale_factor)

        # Ограничение максимального размера графика
        fig_width = min(fig_width, 20)
        fig_height = min(fig_height, 16)

        # Создание фигуры с рассчитанными размерами
        plt.figure(figsize=(fig_width, fig_height), dpi=100)

        # Построение тепловой карты
        sns.heatmap(data, 
                    annot=True,        # Добавление числовых аннотаций
                    cmap="coolwarm",   # Сине-красная цветовая схема
                    fmt='.2f')         # Формат чисел: 2 знака после запятой

        # Сохранение графика если указано имя файла
        if save_name is not None and save_name != "":
            plt.savefig(save_name)
            
        # Отображение графика
        plt.show()
        
    except Exception as e:
        print(f"Исключение при построении тепловой карты: {e}")


def main() -> None:
    '''
    Основная функция программы.
    '''
    # Загрузка набора данных
    df = get_dataset(URL, COLUMNS)
    
    # Вычисление матрицы корреляции
    correlation_matrix = calculate_correlation(df)
    
    # Построение тепловой карты с сохранением
    draw_heatmap_with_coolwarm(correlation_matrix, "corr_mat.png")


if __name__ == "__main__":
    main()