

def run_query_and_inspect(query, query_engine, show_nodes=True):
    """
    Ejecuta una query sobre un query_engine de LlamaIndex y muestra la respuesta generada 
    junto con los chunks recuperados (nodos fuente), incluyendo su puntuación y contenido.

    Args:
        query (str): Pregunta que se lanza al sistema RAG.
        query_engine (RetrieverQueryEngine): Motor de consulta configurado con un retriever.
        show_nodes (bool): Si se deben imprimir los nodos fuente recuperados. Default: True.

    Returns:
        Response: Objeto `response` generado por el motor de consulta, con acceso a la respuesta y nodos fuente.
    """
    response = query_engine.query(query)

    print(f"\nConsulta: {query}\n{'='*60}")
    print(f"\nRespuesta: {response.response}\n{'='*60}")

    if show_nodes:
        for i, node in enumerate(response.source_nodes):
            print(f"\n--- Nodo {i+1} ---")
            print(f"Score: {node.score:.4f}")
            print(node.node.get_content())

    return response

def display_question_result(result):
    """
    Imprime en consola los resultados de una predicción realizada por el modelo para una pregunta tipo test.

    Args:
        result (dict): Resultado generado por `ask_question_with_options`.

    Output:
        Información legible en consola sobre:
            - Pregunta y opciones
            - Respuesta generada vs correcta
            - Chunks usados para justificar la respuesta
    """
    print("ID Pregunta:", result["question_id"])
    print("\nPregunta:\n", result["question"])
    print("\nOpciones:\n", result["options"])
    print("\nAcertó?:", result["is_correct"])
    print("\nRespuesta generada por el modelo:\n", result["predicted_answer"])
    print("\nRespuesta correcta:", result["correct_answer"])

    if "response_obj" in result:
        print(f"\n🔎 Chunks usados para responder la pregunta {result['question_id']}:\n")
        for i, node in enumerate(result["response_obj"].source_nodes):
            print(f"--- Chunk #{i+1} | Score: {node.score:.4f} ---")
            print(node.node.get_content().strip())
            print()