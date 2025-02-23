import csv
import os
import json
import time
import gc
from typing import List, Dict, Any
from tqdm import tqdm

from utils.llm import Ollama
from utils import config
from agent.agent import Agent

def read_questions(file_path: str) -> List[str]:
    """Legge le domande da un file CSV."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return [row['domanda'] for row in reader]
    except Exception as e:
        print(f"Error reading questions file: {e}")
        return []

def load_dish_mapping(file_path: str) -> Dict[str, Any]:
    """Carica il mapping dei piatti da un file JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading dish mapping file: {e}")
        return {}

def load_existing_results(output_file: str) -> Dict[int, str]:
    """
    Carica i risultati già presenti nel file di output (se esistente) in un dizionario con chiave row_id.
    """
    existing = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    try:
                        row_id = int(row['row_id'])
                        existing[row_id] = row['result']
                    except ValueError:
                        continue
        except Exception as e:
            print(f"Error reading existing output: {e}")
    return existing

def write_results_file(output_file: str, results: Dict[int, str], total_questions: int) -> None:
    """
    Scrive il file CSV con una riga per ogni domanda, usando l'indice della domanda.
    Se per una domanda non esiste un risultato, scrive una stringa vuota.
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['row_id', 'result']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_id in range(1, total_questions + 1):
            writer.writerow({'row_id': row_id, 'result': results.get(row_id, '')})

def process_and_write_questions(questions: List[str],
                                output_file: str,
                                dish_mapping: Dict[str, Any],
                                repeat_empty: bool = True) -> None:
    """
    Per ogni domanda:
      - Se la domanda esiste già nel file di output con una risposta non vuota, la salta.
      - Altrimenti, la elabora.
      - Dopo ogni risposta, il file CSV viene aggiornato per mantenere l'ordine corretto.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Carica i risultati esistenti (se presenti)
    existing_results = load_existing_results(output_file)
    results = existing_results.copy()
    total_questions = len(questions)

    structured_prompt = """
    Rispondi con SOLO una lista di ID numerici separati da virgola. 
    Ad esempio: 12,34,56
    Non aggiungere testo aggiuntivo. Solo numeri e virgole.
    """

    for i, question in enumerate(tqdm(questions, desc="Processing questions"), 1):
        # Se la domanda è già stata processata con una risposta non vuota, la saltiamo.
        if i in results and results[i].strip():
            continue

        # Rimuove eventuali virgolette dalla domanda
        question = question.replace("\"", "")
        print(f"\nProcessing question {i}: {question}")

        # Crea una nuova istanza dell'agente ad ogni iterazione
        llm_instance = Ollama(config.MODEL_NAME)
        agent = Agent(llm=llm_instance, verbose=True)

        final_answer = agent.run(question, structured_prompt).strip()
        print(f"Risposta: {final_answer}")

        results[i] = final_answer

        # Aggiorna subito il file CSV dopo ogni risposta
        write_results_file(output_file, results, total_questions)

        del agent
        del llm_instance
        gc.collect()

        time.sleep(2)

    print(f"\nResults written to {output_file}")

def main():
    input_file = os.path.join(config.DATASET_PATH, 'domande.csv')
    output_file = os.path.join(config.OUTPUT_PATH, 'results.csv')
    dish_mapping_file = os.path.join(config.DATASET_PATH, 'Misc', 'dish_mapping.json')

    questions = read_questions(input_file)
    dish_mapping = load_dish_mapping(dish_mapping_file)

    # Processa le domande: se la risposta è vuota o la domanda non esiste, viene elaborata.
    process_and_write_questions(questions, output_file, dish_mapping, repeat_empty=True)

if __name__ == "__main__":
    main()
