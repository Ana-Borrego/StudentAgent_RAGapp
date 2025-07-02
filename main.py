import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, get_response_synthesizer, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings

# Set up --------------------------------------------------------------------------------------------------
load_dotenv()

# Config LLM w Prompts --------------------------------------------------------------------------------------------------
system_prompt = """
Eres un experto en PySpark y Spark, especializado en el temario de la certificación Databricks Certified Associate Developer for Apache Spark.

Debes responder única y exclusivamente utilizando la información que se encuentra en los documentos proporcionados como contexto. 

Si la respuesta a la pregunta no se obtiene claramente del contexto, intenta razonar con el contexto una respuesta útil para responder a la consulta.

Si la respuesta a la pregunta no se puede obtener del contexto ni se puede razonar a partir de él, responde exactamente: "No he encontrado información al respecto."
"""

qa_template_str = """
Contexto: 
{context_str}

Utiliza exclusivamente la información anterior para responder. 

Pregunta: {query_str}

Si no encuentras suficiente información para dar una respuesta clara, responde: 'No he encontrado información al respecto.'

Respuesta:
"""

qa_template = PromptTemplate(qa_template_str)

llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0,
    system_prompt=system_prompt
)

Settings.llm = llm
# Set embedding model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Load Vector index --------------------------------------------------------------------------------------------------
storage_path = "data/index_storage/"
storage_context = StorageContext.from_defaults(persist_dir=storage_path)
index = load_index_from_storage(storage_context)

# Retriever & Query Engine --------------------------------------------------------------------------------------------------
retriever = index.as_retriever(similarity_top_k=5)

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
    llm=Settings.llm,
    text_qa_template=qa_template
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
)

# Use in terminal --------------------------------------------------------------------------------------------------
def main():
    print(" --- Asistente RAG en PySpark. --- ")
    print("Escribe la pregunta o 'exit' para terminar.\n")
    
    while True:
        user_input = input("Escribe tu consulta o 'exit': ")
        if user_input.strip().lower() in ["exit", "salir"]:
            print("Cerrando el asistente.")
            break
        
        try: # trata de hacer la consulta al modelo RAG    
            response = query_engine.query(user_input)
            print(f"Respuesta del Asistente:\n{response.response}\n")
        except Exception as e:
            print(f"Error: {e}\n")
            
if __name__ == "__main__":
    main()