import re
import fitz
import pandas as pd
import itertools
from llama_index import (
    VectorStoreIndex,
    SimpleNodeParser,
    Document,
    StorageContext,
    SimpleDocumentStore,
    SimpleIndexStore,
    SimpleVectorStore,
    Settings
)
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.prompts import QueryBundle
from llama_index.response_synthesizers import get_response_synthesizer

## PREPROCESS DATA ############################################################################################
def delete_lines_w_digits_only(text):
    """
    Elimina líneas que contienen exclusivamente números (como números de página).

    Args:
        text (str): Texto completo del que se eliminarán las líneas con solo dígitos.

    Returns:
        str: Texto sin líneas que consistan únicamente en números.
    """
    pattern = r'(?m)^\s*\d+\s*$'
    return re.sub(pattern, '', text)

def delete_strange_characters(text):
    """
    Elimina caracteres Unicode no deseados, como el espacio estrecho (U+202F), 
    que suele aparecer al copiar bloques de código desde PDF.

    Args:
        text (str): Texto que puede contener caracteres especiales no deseados.

    Returns:
        str: Texto con los caracteres no deseados eliminados.
    """
    return re.sub("\u202f", "", text)

def remove_trailing_section_by_keywords(text, keywords=("Sample questions", "Sample question", "Answer")):
    """
    Elimina todo el contenido del texto desde que aparece una palabra clave 
    (por ejemplo: "Sample questions", "Answer") hasta el final del texto.

    Args:
        text (str): Texto a limpiar.
        keywords (tuple): Palabras clave que indican el inicio de una sección no deseada.

    Returns:
        str: Texto truncado, sin la sección posterior a las palabras clave.
    """
    lines = text.splitlines()
    lower_keywords = [kw.lower() for kw in keywords]
    
    for i, line in enumerate(lines):
        if line.strip().lower() in lower_keywords:
            return "\n".join(lines[:i]).strip()
    return text

def extract_range_of_pages(data_path, start_page=31, end_page=208, add_pages_sep=True):
    """
    Extrae el texto de un rango de páginas de un PDF, aplicando limpieza de 
    líneas numéricas y secciones irrelevantes como preguntas de ejemplo.

    Args:
        data_path (str): Ruta al archivo PDF.
        start_page (int): Página de inicio (base 0).
        end_page (int): Página final (inclusiva).
        add_pages_sep (bool): Si se debe insertar un marcador de separación entre páginas.

    Returns:
        tuple[str, fitz.Document]: Texto concatenado limpio y el objeto `fitz.Document` original.
    """
    doc = fitz.open(data_path)
    selected_text = ""

    for n_page in range(start_page, end_page + 1):
        page = doc.load_page(n_page)
        text_from_page = page.get_text()

        # Limpieza
        cleaned_text = delete_lines_w_digits_only(text_from_page)
        cleaned_text = delete_strange_characters(cleaned_text)
        cleaned_text = remove_trailing_section_by_keywords(cleaned_text)
        
        if cleaned_text.strip():
            if add_pages_sep:
                selected_text += f"\n--- New page {n_page} ---\n"
            selected_text += cleaned_text + "\n"

    return selected_text, doc

## SCORE_DF_RETRIEVER ###########################################################################################

def get_responses_similarity(query_str, query_engine, top_k):
    """
    Ejecuta una consulta contra un `query_engine` y extrae los nodos fuente devueltos,
    incluyendo su contenido, puntuación de similitud y posición.

    Args:
        query_str (str): Pregunta o consulta a evaluar.
        query_engine (RetrieverQueryEngine): Motor de consulta LlamaIndex con acceso a los nodos.
        top_k (int): Número de nodos relevantes a recuperar.

    Returns:
        dict: Diccionario con la siguiente información:
            - "query": query repetida `top_k` veces
            - "response": respuesta del LLM repetida `top_k` veces
            - "position_node": posición del nodo en el ranking
            - "node_content": contenido del nodo
            - "node_score": puntuación de similitud del nodo
    """
    response = query_engine.query(query_str)

    info_nodes = {
        "query": [query_str] * top_k,
        "position_node": [],
        "node_content": [],
        "node_score": [],
        "response": [response] * top_k
    }
    for i, node_with_score in enumerate(response.source_nodes):
        info_nodes["node_content"].append(node_with_score.node.get_content())
        info_nodes["node_score"].append(node_with_score.score)
        info_nodes["position_node"].append(i + 1)

    return info_nodes


def create_score_DataFrame(
    content_text,
    chunk_sizes,
    chunk_overlap=20,
    top_k=10,
    save_storage=False,
    load_storage=False,
    storage_path="../data/index_storage/",
    queries=[]
):
    """
    Evalúa el rendimiento del sistema RAG para diferentes valores de `chunk_size` y genera
    un DataFrame con las puntuaciones de similitud obtenidas en las queries.

    Args:
        content_text (str): Texto base (por ejemplo, el libro ya limpiado) sobre el que generar los chunks.
        chunk_sizes (list[int]): Lista de tamaños de chunk que se quieren probar.
        chunk_overlap (int, optional): Número de tokens de solapamiento entre chunks. Default: 20.
        top_k (int, optional): Número de nodos a recuperar por consulta. Default: 10.
        save_storage (bool, optional): Si se desea guardar el index vectorial en disco. Default: False.
        load_storage (bool, optional): Si se desea cargar un index previamente guardado. Default: False.
        storage_path (str, optional): Ruta donde guardar/cargar el index. Default: "../data/index_storage/".
        queries (list[str]): Lista de preguntas a realizar sobre el contenido.

    Returns:
        pd.DataFrame: DataFrame con todas las combinaciones evaluadas, incluyendo:
            - query, response
            - chunk_size, chunk_overlap
            - position_node, node_content, node_score
    """
    pdf_doc = Document(text=content_text.strip())
    all_rows = []

    for chunk_size in chunk_sizes:
        parser = SimpleNodeParser().from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = parser.get_nodes_from_documents([pdf_doc])

        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        index = VectorStoreIndex(nodes)

        if save_storage:
            index.storage_context.persist(persist_dir=storage_path)
        if load_storage:
            storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore.from_persist_dir(persist_dir=storage_path),
                vector_store=SimpleVectorStore.from_persist_dir(persist_dir=storage_path),
                index_store=SimpleIndexStore.from_persist_dir(persist_dir=storage_path),
            )

        retriever = index.as_retriever(similarity_top_k=top_k)
        query_engine_VSI = RetrieverQueryEngine.from_args(retriever=retriever)

        for query in queries:
            query_info = get_responses_similarity(query, query_engine=query_engine_VSI, top_k=top_k)
            query_info["chunk_size"] = [chunk_size] * top_k
            query_info["chunk_overlap"] = [chunk_overlap] * top_k
            chunk_df = pd.DataFrame(query_info)
            all_rows.append(chunk_df)

    score_df = pd.concat(all_rows, ignore_index=True)
    return score_df

## Version 2 de las funciones de evaluación

def create_score_DataFrame_v2(
    content_text,
    chunk_sizes,
    chunk_overlaps,
    top_k=10,
    save_storage=False,
    load_storage=False,
    storage_path="../data/index_storage/",
    queries=[]
):
    """
    Evalúa el rendimiento del sistema RAG para combinaciones de `chunk_size` y `chunk_overlap`,
    y devuelve un DataFrame con las puntuaciones de similitud obtenidas.

    Args:
        content_text (str): Texto base para generar los chunks.
        chunk_sizes (list[int]): Lista de tamaños de chunk a probar.
        chunk_overlaps (list[int]): Lista de valores de solapamiento entre chunks.
        top_k (int): Número de nodos relevantes a recuperar.
        save_storage (bool): Si se debe guardar el índice vectorial.
        load_storage (bool): Si se debe cargar desde almacenamiento existente.
        storage_path (str): Ruta al almacenamiento en disco.
        queries (list[str]): Preguntas para evaluar el índice.

    Returns:
        pd.DataFrame: Resultados de las consultas con columnas:
            - query, response, position_node, node_content, node_score
            - chunk_size, chunk_overlap
    """
    pdf_doc = Document(text=content_text.strip())
    all_rows = []
    combinations = list(itertools.product(chunk_sizes, chunk_overlaps))

    for chunk_size, chunk_overlap in combinations:
        parser = SimpleNodeParser().from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = parser.get_nodes_from_documents([pdf_doc])

        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        index = VectorStoreIndex(nodes)

        if save_storage:
            index.storage_context.persist(persist_dir=storage_path)
        if load_storage:
            storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore.from_persist_dir(persist_dir=storage_path),
                vector_store=SimpleVectorStore.from_persist_dir(persist_dir=storage_path),
                index_store=SimpleIndexStore.from_persist_dir(persist_dir=storage_path),
            )

        retriever = index.as_retriever(similarity_top_k=top_k)
        query_engine_VSI = RetrieverQueryEngine.from_args(retriever=retriever)

        for query in queries:
            query_info = get_responses_similarity(query, query_engine=query_engine_VSI, top_k=top_k)
            query_info["chunk_size"] = [chunk_size] * top_k
            query_info["chunk_overlap"] = [chunk_overlap] * top_k
            chunk_df = pd.DataFrame(query_info)
            all_rows.append(chunk_df)

    score_df = pd.concat(all_rows, ignore_index=True)
    return score_df


def analyze_score_retriever_v2(df):
    """
    Calcula métricas agregadas de similitud (node_score) para evaluar chunks.

    Agrupa los resultados por `chunk_size` y `chunk_overlap`, y calcula:
    - La media de node_score en todas las posiciones.
    - La media de node_score únicamente para el nodo en posición 1.

    Args:
        df (pd.DataFrame): DataFrame generado por `create_score_DataFrame_v2`.

    Returns:
        pd.DataFrame: Métricas agregadas ordenadas por chunk_size y chunk_overlap.
    """
    agg1 = df.groupby(["chunk_size", "chunk_overlap"])["node_score"].mean().reset_index()
    agg1.columns = ["chunk_size", "chunk_overlap", "mean_score_all_positions"]

    agg2 = df[df["position_node"] == 1].groupby(["chunk_size", "chunk_overlap"])["node_score"].mean().reset_index()
    agg2.columns = ["chunk_size", "chunk_overlap", "mean_score_top1"]

    result = pd.merge(agg1, agg2, on=["chunk_size", "chunk_overlap"])
    return result.sort_values(["chunk_size", "chunk_overlap"]).reset_index(drop=True)

## TESTS ###########################################################################################

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

## EXTRACT EXAMS ###########################################################################################

def extract_cleaned_text(data_path, start_page, end_page):
    """
    Extrae texto limpio de un PDF entre dos páginas, eliminando ruido y estructuras repetitivas.

    Args:
        data_path (str): Ruta al archivo PDF.
        start_page (int): Página inicial (base 0).
        end_page (int): Página final (inclusive).

    Returns:
        str: Texto limpio concatenado de las páginas indicadas.
    """
    doc = fitz.open(data_path)
    selected_text = ""

    for n_page in range(start_page, end_page + 1):
        page = doc.load_page(n_page)
        text_from_page = page.get_text()

        cleaned_text = delete_lines_w_digits_only(text_from_page)
        cleaned_text = delete_strange_characters(cleaned_text)
        cleaned_text = re.sub(r"(Mock Test\s*\d+|Questions)", "", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"\n{2,}", "\n", cleaned_text)

        if cleaned_text.strip():
            selected_text += cleaned_text + "\n"

    return selected_text


def extract_options(text_block):
    """
    Extrae las opciones de respuesta tipo A. B. C. D. E. desde un bloque de texto.

    Args:
        text_block (str): Bloque de texto con opciones.

    Returns:
        dict: Diccionario con claves 'A' a 'E' y sus respectivos textos.
    """
    split = re.split(r'\n(?=([A-E])\.\s)', text_block.strip())
    results = {}
    current_key = None
    buffer = []

    for part in split:
        m = re.match(r'^([A-E])\.\s+(.*)', part, re.DOTALL)
        if m:
            if current_key:
                results[current_key] = "\n".join(buffer).strip()
            current_key = m.group(1)
            buffer = [m.group(2)]
        elif current_key:
            buffer.append(part)

    if current_key:
        results[current_key] = "\n".join(buffer).strip()

    return results


def extract_questions_from_text(cleaned_text: str, test_number: int):
    """
    Extrae preguntas tipo test desde texto limpio, incluyendo enunciado, opciones y respuesta correcta.

    Args:
        cleaned_text (str): Texto limpio con preguntas y respuestas.
        test_number (int): Número de test (Mock Test), usado para el ID.

    Returns:
        list[dict]: Lista de preguntas con campos:
            - question_id
            - question
            - options
            - correct_answer
    """
    pattern = re.compile(
        r"^Question\s+(\d+):\s*((?:.*(?:\n(?!Question\s+\d+:|\nAnswers|\Z).*)*))",
        re.MULTILINE
    )

    answers_pattern = re.compile(r"\n\s*(\d+)\.\s*([A-E])")
    answers = {int(q): a for q, a in answers_pattern.findall(cleaned_text)}

    questions = []
    for match in pattern.finditer(cleaned_text):
        q_num = int(match.group(1))
        q_text = match.group(2).strip()

        options = extract_options(q_text)

        if options:
            option_a_match = re.search(r"\nA\.\s", q_text)
            if option_a_match:
                question_text = q_text[:option_a_match.start()].strip()
            else:
                question_text = q_text.strip()
        else:
            question_text = q_text.strip()

        question_id = f"Test_{test_number}_{q_num}"
        correct = answers.get(q_num, None)

        questions.append({
            "question_id": question_id,
            "question": question_text,
            "options": options,
            "correct_answer": correct
        })

    return questions

## TESTS ON EXAMS ##################################################################################################
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
