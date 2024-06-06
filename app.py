from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

from load_file import name_columns

app = Flask(__name__)

# Загрузка модели и нормализатора
model = load_model('notebooks/composite_model.h5')
scaler = joblib.load('notebooks/scaler.pkl')

path_first = 'data/X_bp.xlsx'
path_second = 'data/X_nup.xlsx'

# Название колонн
columns = name_columns(path_first, path_second)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Получение данных от пользователя
        user_data = {}
        for column in columns:
            user_data[column] = float(request.form[column])
        
        # Создание массива признаков
        X_new = np.array([[user_data[column] for column in columns]])
        
        # Нормализация данных
        X_new_scaled = scaler.transform(X_new)
        
        # Предсказание
        predictions = model.predict(X_new_scaled)
        
        matrix_prediction = predictions[0]

        
        return render_template('index.html', columns=columns, matrix=matrix_prediction)
    return render_template('index.html', columns=columns)
    

if __name__ == '__main__':
    app.run(debug=True, port=5001)
