import re
import fitz  # PyMuPDF

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