from dotenv import load_dotenv
from llama_index.core import StorageContext, get_response_synthesizer, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
import json
import scripts.question_prompting as evaluation_functions

# Configuración --------------------------------------------------------------------------------------------------

def configurar_query_engine(storage_path="data/index_storage/") -> RetrieverQueryEngine:
    load_dotenv()

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

    llm = OpenAI(model="gpt-4o-mini", temperature=0, system_prompt=system_prompt)
    Settings.llm = llm
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    storage_context = StorageContext.from_defaults(persist_dir=storage_path)
    index = load_index_from_storage(storage_context)

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

    return query_engine

def cargar_preguntas_desde_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        preguntas = json.load(f)
    return preguntas

# Use in terminal --------------------------------------------------------------------------------------------------
def main():
    query_engine = configurar_query_engine()

    print(" --- Asistente RAG en PySpark. --- ")
    
    while True:
        print("\nSelecciona una opción:")
        print("1. Preguntar al asistente")
        print("2. Evaluar automáticamente un test")
        print("3. Salir")
        print()
        print("Para salir en cualquier momento escribe 'exit' o 'salir'.")

        option = input("\nIntroduce 1, 2 o 3: ").strip().lower()

        if option == "1":
            while True:
                user_input = input("\nEscribe tu consulta o 'menu' para cambiar de opción: ")
                if user_input.strip().lower() in ["exit", "salir"]:
                    print("Cerrando el asistente.")
                    return
                if user_input.strip().lower() == "menu":
                    break
                try:
                    response = query_engine.query(user_input)
                    print(f"\nRespuesta del Asistente:\n{response.response}\n")
                except Exception as e:
                    print(f"\nError: {e}\n")

        elif option == "2":
            try:
                json_path = "data/exams_test2/clean_questions/questions_all_tests.json"
                preguntas = cargar_preguntas_desde_json(json_path)
                test_num = input("Selecciona el test (1 o 2): ").strip()
                
                if test_num not in ["1", "2"]:
                    print("Número de test no válido.")
                    continue
                
                evaluation_functions.ejecutar_test_automatico(query_engine, preguntas, test_exam=int(test_num))
            
            except FileNotFoundError:
                print("❌ No se encontró el archivo JSON con las preguntas.")
            except Exception as e:
                print(f"❌ Error durante la evaluación: {e}")

        elif option == "3" or option in ["exit", "salir"]:
            print("Cerrando el asistente.")
            break

        else:
            print("Opción no válida. Por favor, selecciona 1, 2 o 3.")  
            
if __name__ == "__main__":
    main()