from flask import Flask, render_template, request
from datetime import datetime
import re
import os
import pandas as pd
import ModeloRegresion as modelo 
import LinearRegression

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()

    match_object = re.match("[a-zA-Z]+", name)

    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "Friend"

    content = "Hello there, " + clean_name + "! Hour: " + str(now)
    return content

@app.route("/menu")
def menu():
    return render_template("Menu.html") # menu de navegacion

@app.route("/pagina")
def pagina():
    return render_template("pagina.html")

@app.route("/LinearRegression", methods= ["GET", "POST"])
def inicio():
    prediccion = None
    if request.method == "POST":
     try:
        edad = float(request.form["edad"])
        prediccion = LinearRegression.calcularCosto(edad)
     except ValueError:
        prediccion = "Datos no validos"
        
    LinearRegression.generar_grafico()
    return render_template("LinearRegressionGrades.html", result = prediccion)

@app.route('/RegresionLogistica')
def RegresionLogistica():
    return render_template('RegresionLogistica.html') #corregir hoooooooooooy :(

#Modelo de Regresión Logística

@app.route('/desercion', methods=['POST', 'GET'])
def desercion():
    if request.method == 'POST':
        # Obtener datos del formulario
        datos = {
            'notas': float(request.form.get('notas')),
            'asistencia': float(request.form.get('asistencia')),
            'participacion': int(request.form.get('participacion')),
            'tipo_colegio': 0 if request.form.get('tipo_colegio') == 'Publico' else 1
        }
        
        # Realizar predicción usando el modelo
        resultado = modelo.predecir_con_matriz(datos)
        
        return render_template('Desercion.html', 
                             datos=datos,
                             probabilidad=resultado['probabilidad'],
                             resultado=resultado['resultado'],
                             accuracy=resultado['accuracy'],
                             precision=resultado['precision'],
                             recall=resultado['recall'])
    else:
        return render_template('Desercion.html', 
                             datos=None, 
                             probabilidad="0%", 
                             resultado="", 
                             accuracy=0.0, 
                             precision=0.0, 
                             recall=0.0)


if __name__ == "__main__":
    app.run(debug=True)