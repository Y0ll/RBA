import numpy as np
import pandas as pd
import ipaddress
from flask import Flask, request, jsonify
import io
import pickle
def ip_to_int(ip):
    return int(ipaddress.ip_address(ip))

app = Flask(__name__)

@app.route('/', methods=['POST'])
def handle_request():
    # Получение данных из запроса
    csv_data = request.data.decode('utf-8')

    # Преобразование CSV-данных в DataFrame
    try:
        data = pd.read_csv(io.StringIO(csv_data))
        data['Login Hour'] = pd.to_datetime(data['Login Timestamp']).dt.hour

        # Converting Booleans To Integers
        data['Login Successful'] = data['Login Successful'].astype(np.uint8)

        # Dropping Unneeded Columns
        data = data.drop(columns=["Round-Trip Time [ms]", 'Region', 'City', 'Login Timestamp', 'index'])

        # Converting Strings To Integers
        data['User Agent String'], _ = pd.factorize(data['User Agent String'])
        data['Browser Name and Version'], _ = pd.factorize(data['Browser Name and Version'])
        data['OS Name and Version'], _ = pd.factorize(data['OS Name and Version'])

        # Converting IP Addresses To Integers
        data['IP Address'] = data['IP Address'].apply(ip_to_int)

        predictions = pipline.predict_proba(data)[:, 1]
        if(predictions <= 1e-05):
            risk_analysis = 'min'
        elif (predictions <= 5e-05 and predictions > 1e-05):
            risk_analysis = 'middle'
        else:
            risk_analysis = 'high'

        response_data = pd.DataFrame({'probability': [risk_analysis]})

        # Преобразование DataFrame обратно в CSV
        response_csv = response_data.to_csv(index=False)

        # Отправка ответа обратно в NiFi
        return response_csv, 200, {'Content-Type': 'text/csv'}

    except Exception as e:
        print("Ошибка при обработке CSV:", e)
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    with open('bin/model.pkl', 'rb') as file:
        pipline = pickle.load(file)
    app.run(host='0.0.0.0', port=9292)

