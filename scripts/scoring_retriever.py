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
