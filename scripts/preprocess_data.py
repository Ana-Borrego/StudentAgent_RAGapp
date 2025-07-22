import re
import fitz #PyMuPDF

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
