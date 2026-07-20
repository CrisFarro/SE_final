import io
import pandas as pd
from flask import Flask, render_template, request, jsonify
from agente import AgenteBayesianoAdquisiciones, procesar_csv_df

app = Flask(__name__)
agente = AgenteBayesianoAdquisiciones()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/evaluar-manual', methods=['POST'])
def evaluar_manual():
    try:
        data = request.json
        presupuesto = float(data.get('presupuesto_limite', 5000))
        plazo_max = int(data.get('plazo_maximo', 15))
        
        datos_proveedor = {
            'name': data.get('name', 'Proveedor Ficticio'),
            'precio': float(data.get('precio', 0)),
            'plazo': int(data.get('plazo', 0)),
            'iso': bool(data.get('iso', False)),
            'historial': bool(data.get('historial', False)), # True = con litigios
            'credito': bool(data.get('credito', False))
        }
        
        resultado = agente.evaluar_propuesta(datos_proveedor, presupuesto, plazo_max)
        return jsonify({"status": "success", "data": [resultado]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/evaluar-csv', methods=['POST'])
def evaluar_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No se subió ningún archivo"}), 400
            
        file = request.files['file']
        presupuesto = float(request.form.get('presupuesto_limite', 5000))
        plazo_max = int(request.form.get('plazo_maximo', 15))
        
        # Leer directamente en memoria sin guardar en disco
        stream = io.StringIO(file.stream.read().decode("utf-8", errors="ignore"))
        df = pd.read_csv(stream)
        
        ofertas = procesar_csv_df(df)
        resultados = [agente.evaluar_propuesta(of, presupuesto, plazo_max) for of in ofertas]
        
        return jsonify({"status": "success", "data": resultados})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)