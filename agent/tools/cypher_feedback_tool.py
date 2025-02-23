import json
import os
from agent.tools.base import tool

@tool
def cypher_feedback_tool(user_question: str = ""):
    """
    Funzione per raccogliere il feedback dall'utente, includendo la domanda in linguaggio naturale e la query Cypher.
    Memorizza il feedback nel file JSON per migliorare le query in futuro.
    Usa questo tool SOLO se cypher_search_tool risponde "Nessun piatto trovato".
    
    ** TOOL PRIORITY: 2
    """
    feedback_query = input("Inserisci una query Cypher corretta: ")
    if not feedback_query or not user_question:
        return "Errore: Query Cypher o domanda in linguaggio naturale non forniti."
    
    # Carica i feedback già esistenti
    feedback_file = "feedback.json"
    feedback_store = load_feedback(feedback_file)
    
    # Crea un nuovo feedback
    new_feedback = {
        "user_question": user_question,
        "cypher_query": feedback_query
    }
    
    # Aggiungi o aggiorna il feedback nel store
    feedback_store = save_feedback(feedback_store, feedback_file, new_feedback)
    return "Feedback salvato correttamente. \n Ora puoi ripetere la Cypher query con il tool dedicato."

def save_feedback(feedback_store, feedback_file, new_feedback):
    """
    Salva i feedback raccolti nel file JSON, evitando duplicati per la stessa domanda.
    Se una domanda è già presente, aggiorna il feedback esistente.
    """
    try:
        # Controlla se esiste già un feedback per la domanda
        question_exists = False
        for feedback_id, feedback in feedback_store.items():
            if feedback["user_question"].strip().lower() == new_feedback["user_question"].strip().lower():  # Case-insensitive comparison
                # Se esiste, aggiorna la query Cypher
                feedback_store[feedback_id]["cypher_query"] = new_feedback["cypher_query"]
                question_exists = True
                print(f"Feedback aggiornato per la domanda: {new_feedback['user_question']}")
                break

        # Se non esiste un feedback, aggiungi il nuovo feedback
        if not question_exists:
            feedback_id = len(feedback_store) + 1
            feedback_store[feedback_id] = {
                "user_question": new_feedback["user_question"],
                "cypher_query": new_feedback["cypher_query"]
            }
            print(f"Nuovo feedback salvato per la domanda: {new_feedback['user_question']}")

        # Salva i feedback nel file JSON
        with open(feedback_file, "w") as f:
            json.dump(feedback_store, f, indent=4)
        print(f"Feedback memorizzato nel file {feedback_file}")
        
        return feedback_store
        
    except Exception as e:
        print(f"Errore durante il salvataggio del feedback: {e}")
        return feedback_store  # Return the feedback store as it is in case of error

def load_feedback(feedback_file):
    """
    Carica i feedback precedentemente memorizzati dal file JSON.
    """
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Errore durante il caricamento del feedback: {e}")
            return {}
    return {}