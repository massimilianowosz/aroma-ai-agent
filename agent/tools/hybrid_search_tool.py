from agent.tools.base import tool
from utils.logger import Logger
from rag.searcher.cypher import CypherRAGRetriever
from rag.searcher.embedding import EmbeddingRAGRetriever

@tool
def hybrid_search_tool(query: str, question: str="") -> str:
    """
    Esegue una ricerca ibrida nel database Neo4j utilizzando sia una query Cypher
    sia una ricerca per embedding. Se la ricerca Cypher non restituisce risultati,
    viene usato il fallback con la ricerca per embedding. Se entrambi forniscono risultati,
    questi vengono combinati e riordinati (re-ranking) in base a un punteggio combinato.
    
    Args:
        query (str): La domanda per cui effettuare la ricerca.
        question (str): Full user question unfiltered.
        
    Returns:
        str: Risultato della ricerca ibrida formattato in output.
    """
    logger = Logger()
    if not question:
        question = query
        
    # --- Ricerca tramite Cypher ---
    cypher_retriever = CypherRAGRetriever()
    cypher_response = cypher_retriever.answer_question(query)
    cypher_results = cypher_response.get('results', [])
    cypher_retriever.close()
    
    extracted_ids_cypher = []
    
    for record in cypher_results:
        if "p" in record:
            properties = record['p']._properties  # Otteniamo le proprietà del nodo 'p'
            nome = properties.get('nome', 'Unknown')
            id_val = properties.get('dish_mapping_id', 0)
            score = 1.0  # Default score per Cypher
        
        elif any(k.startswith("p.") for k in record.keys()):
            id_val = record.get("p.dish_mapping_id", 0)
            nome = record.get("p.nome", "Unknown")
            score = 1.0  # Default score per Cypher
        
        else:
            id_val = next((v for v in record.values() if isinstance(v, int) and 1 <= v <= 286), 0)
            nome = "Unknown"
            score = 1.0  # Default score per Cypher
        
        if 1 <= id_val <= 286:
            extracted_ids_cypher.append({"id": id_val, "nome": nome, "score": score})
    
    logger.info("CYPHER")
    logger.info(extracted_ids_cypher)
    
    # --- Ricerca tramite embedding ---
    embedding_retriever = EmbeddingRAGRetriever()
    embedding_response = embedding_retriever.answer_question(question, threshold=0.3)
    embedding_results = embedding_response.get('results', [])
    if embedding_results:
        embedding_entities = embedding_results[0].get('entities', [])
    else:
        embedding_entities = []
    
    logger.info("EMBEDDING")
    logger.info(embedding_entities)
    
    extracted_ids_embedding = []
    for record in embedding_entities:
        id_val = record.get('id', 0)
        nome = record.get('nome', 'Unknown')
        score = record.get('score', 0.0)
        
        if 1 <= id_val <= 286:
            extracted_ids_embedding.append({"id": id_val, "nome": nome, "score": score})
    
    # --- Combinazione e re-ranking ---
    combined_dict = {}
    
    for rec in extracted_ids_cypher:
        combined_dict[rec["id"]] = {"nome": rec["nome"], "score": rec["score"]}
    
    for rec in extracted_ids_embedding:
        if rec["id"] in combined_dict:
            combined_dict[rec["id"]]["score"] = (combined_dict[rec["id"]]["score"] + rec["score"]) / 2
        else:
            combined_dict[rec["id"]] = {"nome": rec["nome"], "score": rec["score"]}
    
    combined = sorted(
        [{"id": rec_id, "nome": data["nome"], "score": data["score"]} for rec_id, data in combined_dict.items()],
        key=lambda x: x["score"],
        reverse=True
    )
    
    # --- Costruzione dell'output ---
    if combined:
        output_lines = [str(rec['id']) for rec in combined]
        output_str = ",".join(output_lines)
    else:
        output_str = "1"
    
    return output_str