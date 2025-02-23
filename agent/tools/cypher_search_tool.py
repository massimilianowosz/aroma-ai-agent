from agent.tools.base import tool
from utils.logger import Logger
from rag.searcher.cypher import CypherRAGRetriever

@tool
def cypher_search_tool(query: str) -> str:
    """
    Interroga il database Neo4j utilizzando la domanda utente per generare ed eseguire una query Cypher.
    Fornire solo la domanda e non una query Cypher.
    ** TOOL PRIORITY: 1
    
    Se il risultato è 'Nessun piatto trovato' l'agente DEVE passare al tool 'cypher_feedback_tool' (se disponibile) per chiedere all'utente una query Cypher corretta."
        
    Args:
        query (str): La domanda da utilizzare per generare la query Cypher e interrogare il database. NO query Cypher.
        
    Returns:
        str: Risultato della query formattato in JSON.
    """
    
    # Crea un'istanza del retriever Neo4j
    retriever = CypherRAGRetriever()
    
    # Esegui il metodo answer_question che genera, esegue la query e restituisce il risultato
    response = retriever.answer_question(query)
    logger = Logger()
    logger.observation(response.get('query'))

    extracted_ids = []
    for record in response.get('results', []):
        if isinstance(record, dict):
            # Case when "p" key exists in the record and it's a dictionary
            if "p" in record:  
                properties = record['p']._properties if hasattr(record['p'], '_properties') else None
                if not properties:
                    properties = getattr(record['p'], 'properties', None)

                # Fallback in case properties or required fields are missing
                nome = properties.get('nome', 'Sconosciuto') if properties else 'Sconosciuto'
                id_val = properties.get('dish_mapping_id', 0) if properties else 0
                if not (1 <= id_val <= 286):
                    id_val = 0

            # Case where the property names are prefixed with "p."
            elif any(k.startswith("p.") for k in record.keys()):
                id_val = record.get("p.dish_mapping_id", 0)
                nome = record.get("p.nome", "Sconosciuto")
                if not (1 <= id_val <= 286):
                    id_val = 0

            else:
                # Try to find a valid integer (1-286) in any value of the record
                id_val = next((v for v in record.values() if isinstance(v, int) and 1 <= v <= 286), 0)

        # Additional fallback in case `record` is not a dictionary
        else:
            id_val = 0
            nome = 'Sconosciuto'
            
            
        extracted_ids.append(f"{id_val}")                  
        
    # Chiudi la connessione a Neo4j
    retriever.close()

    if extracted_ids:
        ids = "\n".join(extracted_ids)
        output_str = f"{ids}" 
    else:
        output_str = "1"
    
    return output_str
