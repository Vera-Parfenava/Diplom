import numpy as np
import pandas as pd

def name_columns(path_first, path_second):

    data1 = pd.read_excel(path_first)
    data2 = pd.read_excel(path_second)

    merged_data = pd.merge(data1, data2, left_index=True, right_index=True, how='inner')

    columns_to_drop = ['Unnamed: 0_x', 'Unnamed: 0_y', 'Соотношение матрица-наполнитель']
    data = merged_data.drop(columns=columns_to_drop)

    columns = [column.title() for column in data.columns]

    return columns
