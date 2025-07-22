import re
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.prompts import QueryBundle
from llama_index.response_synthesizers import get_response_synthesizer

def format_options(options_dict):
    """
    Limpia y formatea las opciones de respuesta tipo A, B, C, D, E en un bloque de texto legible.

    Args:
        options_dict (dict): Diccionario con claves 'A' a 'E' y textos como valores.

    Returns:
        str: Opciones formateadas como:
            A. opción A
            B. opción B
            ...
    """
    clean_options = {}
    for key in options_dict:
        value = options_dict[key]
        value = re.sub(r'\n[A-E]\n?', '', value).strip()
        clean_options[key] = value
    return "\n".join([f"{k}. {v}" for k, v in clean_options.items()])

def extract_predicted_letter(response_text):
    """
    Extrae la letra (A, B, C, D o E) desde un texto generado por el modelo,
    si este sigue el formato: 'Respuesta correcta: <LETRA>\nJustificación: ...'

    Args:
        response_text (str): Texto completo generado por el modelo.

    Returns:
        str or None: Letra extraída ('A' a 'E') en mayúsculas si se encuentra, o None si no.
    """
    match = re.search(r"Respuesta\s+correcta\s*:\s*([A-E])", response_text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def ask_question_with_options(question_obj, retriever, response_mode="compact"):
    """
    Lanza una pregunta tipo test al motor RAG utilizando contexto, opciones y una plantilla fija.
    El modelo debe devolver una letra (A-E) y una justificación, en formato estructurado.

    Args:
        question_obj (dict): Pregunta con los siguientes campos:
            - "question" (str): Enunciado de la pregunta
            - "options" (dict): Diccionario de opciones (A-E)
            - "correct_answer" (str): Respuesta correcta (letra A-E)
            - "question_id" (str): ID único de la pregunta
        retriever (BaseRetriever): Retriever de LlamaIndex configurado con los chunks/contexto.
        response_mode (str): Modo de respuesta del modelo. Default = "compact".

    Returns:
        dict: Resultado con las siguientes claves:
            - question_id
            - question
            - options (formateadas)
            - predicted_answer (respuesta generada como string completo)
            - correct_answer (letra esperada)
            - is_correct (bool que compara respuesta generada con la correcta)
            - response_obj (objeto de respuesta completo para inspección)
    """
    question_text = question_obj["question"]
    options_text = format_options(question_obj["options"])

    query_str = f"""
    {question_text}

    Opciones: 
    {options_text}
        
    IMPORTANTE:
        - Usa solo información del contexto.
        - No respondas con conocimientos externos.
        - Aunque sepas la respuesta correcta, si no existe esa información en el contexto devuelve "No existe información"
    
    Responde de la siguiente manera:
    "
    Respuesta correcta: <LETRA>
    Justificación: <JUSTIFICACIÓN>
    "
    """

    query_bundle = QueryBundle(
        query_str=query_str,
        custom_embedding_strs=[question_text, options_text]
    )

    response_synthesizer = get_response_synthesizer(response_mode=response_mode)

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )

    response = query_engine.query(query_bundle)
    predicted_answer = response.response.strip()

    return {
        "question_id": question_obj["question_id"],
        "question": question_obj["question"],
        "options": format_options(question_obj["options"]),
        "predicted_answer": predicted_answer,
        "correct_answer": question_obj["correct_answer"],
        "is_correct": extract_predicted_letter(predicted_answer) == question_obj["correct_answer"],
        "response_obj": response
    }