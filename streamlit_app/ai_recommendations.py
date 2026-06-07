import json
import streamlit as st
from groq import Groq


def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", None)

    if not api_key:
        return None

    return Groq(api_key=api_key)


def build_recommendation_prompt(predicted_class, display_name):
    return f"""
Eres un asistente de cuidado facial.
El modelo de visión detectó esta condición principal: {display_name}.

Devuelve únicamente JSON válido, sin texto adicional.

Estructura exacta:

{{
  "condition": "{display_name}",
  "description": "Descripción breve de la condición.",
  "routine": [
    "Paso 1 de cuidado",
    "Paso 2 de cuidado",
    "Paso 3 de cuidado"
  ],
  "products": [
    {{
      "name": "Nombre real del producto",
      "brand": "Marca",
      "type": "Tipo de producto",
      "price": "Precio aproximado en euros",
      "how_to_use": "Cómo usarlo",
      "warning": "Advertencia breve si aplica"
    }}
  ],
  "medical_note": "Nota médica breve y prudente."
}}

Reglas:
- Recomienda exactamente 3 productos reales.
- Prioriza marcas conocidas: CeraVe, La Roche-Posay, Vichy, The Ordinary, Bioderma, Eucerin.
- No inventes diagnóstico médico.
- No recomiendes medicamentos de prescripción.
- Usa español claro.
"""


def get_ai_recommendations(predicted_class, display_name):
    client = get_groq_client()

    if client is None:
        return None

    prompt = build_recommendation_prompt(
        predicted_class,
        display_name
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            response_format={
                "type": "json_object"
            }
        )

        content = response.choices[0].message.content

        return json.loads(content)

    except Exception as error:
        st.warning(f"No se pudieron generar recomendaciones con IA: {error}")
        return None