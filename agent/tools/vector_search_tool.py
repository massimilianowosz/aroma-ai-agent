from agent.tools.base import tool
from utils.logger import Logger
from rag.searcher.embedding import EmbeddingRAGRetriever

@tool
def vector_search_tool(query: str, threshold: float = 0.5) -> str:
    """
    Esegue una ricerca basata su embedding nel database Neo4j.
    Calcola l'embedding della query e confronta gli embedding dei nodi "Entity".
    Restituisce i risultati ordinati in base al punteggio di similarità.
    ** TOOL PRIORITY: 3
    
    Args:
        query (str): La domanda o il testo della query.
        threshold (float): 
    Returns:
        str: Risultato della ricerca formattato in output.
    """
    #return ''

    logger = Logger()
    retriever = EmbeddingRAGRetriever()
    
    response = retriever.answer_question(query, threshold)
    if response.get('llm_answer'):
        ids = response.get('llm_answer')
        output_str = f"{ids}" 
    else:
        output_str = "1"
    return output_str
