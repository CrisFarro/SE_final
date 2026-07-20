import os
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

class AgenteBayesianoAdquisiciones:
    def __init__(self):
        self.model = DiscreteBayesianNetwork([
            ('Precio', 'Viabilidad'),
            ('Plazo', 'Viabilidad'),
            ('ISO', 'Viabilidad'),
            ('Historial', 'Confianza'),
            ('Credito', 'Confianza'),
            ('Viabilidad', 'Elegibilidad'),
            ('Confianza', 'Elegibilidad')
        ])

        cpd_precio = TabularCPD(variable='Precio', variable_card=2, values=[[0.3], [0.7]])
        cpd_plazo = TabularCPD(variable='Plazo', variable_card=2, values=[[0.2], [0.8]])
        cpd_iso = TabularCPD(variable='ISO', variable_card=2, values=[[0.5], [0.5]])
        cpd_historial = TabularCPD(variable='Historial', variable_card=2, values=[[0.1], [0.9]])
        cpd_credito = TabularCPD(variable='Credito', variable_card=2, values=[[0.4], [0.6]])

        cpd_viabilidad = TabularCPD(
            variable='Viabilidad', variable_card=2,
            evidence=['Precio', 'Plazo', 'ISO'], evidence_card=[2, 2, 2],
            values=[
                [0.99, 0.85, 0.80, 0.50, 0.78, 0.30, 0.25, 0.05],
                [0.01, 0.15, 0.20, 0.50, 0.22, 0.70, 0.75, 0.95]
            ]
        )

        cpd_confianza = TabularCPD(
            variable='Confianza', variable_card=2,
            evidence=['Historial', 'Credito'], evidence_card=[2, 2],
            values=[
                [0.99, 0.90, 0.30, 0.05],
                [0.01, 0.10, 0.70, 0.95]
            ]
        )

        cpd_elegibilidad = TabularCPD(
            variable='Elegibilidad', variable_card=3,
            evidence=['Viabilidad', 'Confianza'], evidence_card=[2, 2],
            values=[
                [0.95, 0.60, 0.70, 0.05],
                [0.04, 0.35, 0.25, 0.15],
                [0.01, 0.05, 0.05, 0.80]
            ]
        )

        self.model.add_cpds(cpd_precio, cpd_plazo, cpd_iso, cpd_historial, cpd_credito, cpd_viabilidad, cpd_confianza, cpd_elegibilidad)
        self.model.check_model()
        self.infer = VariableElimination(self.model)

    def evaluar_propuesta(self, datos, presupuesto_limite=5000.00, plazo_maximo=15):
        evidencia = {
            'Precio': 1 if datos['precio'] <= presupuesto_limite else 0,
            'Plazo': 1 if datos['plazo'] <= plazo_maximo else 0,
            'ISO': 1 if datos['iso'] else 0,
            'Historial': 0 if datos['historial'] else 1,
            'Credito': 1 if datos['credito'] else 0
        }

        resultado = self.infer.query(variables=['Elegibilidad'], evidence=evidencia)
        probs = resultado.values
        best_idx = probs.argmax()
        
        estados = ['RECHAZADO', 'REQUIERE AUDITORÍA', 'ELEGIBLE (APROBADO)']
        colores = ['red', 'yellow', 'green']

        return {
            "name": datos['name'],
            "evidencia": evidencia,
            "probs": {
                "rechazo": round(float(probs[0]) * 100, 2),
                "auditoria": round(float(probs[1]) * 100, 2),
                "elegible": round(float(probs[2]) * 100, 2)
            },
            "dictamen": estados[best_idx],
            "color": colores[best_idx],
            "crudo": datos
        }

def procesar_csv_df(df):
    df.columns = df.columns.str.strip().str.lower()
    column_mapping = {
        'empresa': 'name', 'precio': 'precio', 'plazo': 'plazo',
        'iso': 'iso', 'historial': 'historial', 'credito': 'credito'
    }
    df = df.rename(columns=column_mapping)

    if df['precio'].dtype == object:
        df['precio'] = df['precio'].astype(str).str.replace('$', '', regex=False)
        df['precio'] = df['precio'].str.replace(',', '', regex=False).str.strip()
        df['precio'] = pd.to_numeric(df['precio'])

    df['plazo'] = pd.to_numeric(df['plazo'])

    def parse_boolean(val):
        if pd.isna(val): return False
        val_str = str(val).strip().lower()
        return val_str in ['sí', 'si', 'sí ', 'si ', 'true', '1', '1.0', 'yes', 'y']

    for col in ['iso', 'historial', 'credito']:
        if col in df.columns:
            df[col] = df[col].apply(parse_boolean)

    return df.to_dict(orient='records')