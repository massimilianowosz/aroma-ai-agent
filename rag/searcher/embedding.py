import json
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import database
from utils.llm import Ollama as LLM
from utils import config

class EmbeddingRAGRetriever:
    def __init__(self):
        self.db = database.Neo4jClient()
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm = LLM(config.MODEL_NAME)

    def retrieve_entities(self, query: str, threshold: float = 0.6, top_k: int = 5) -> dict:
        """
        Compute the embedding of the query and compare it with the embeddings
        of "Entity" nodes in the database. Returns a dictionary of results
        (up to top_k) with a similarity score above the threshold.

        Args:
            query (str): The question/query text.
            threshold (float): Minimum similarity score to consider.
            top_k (int): Maximum number of results to return.

        Returns:
            dict: {"results": [ { "id": <id>, "name": <name>, "score": <score>, "text": <text> }, ... ]}
        """
        threshold = float(threshold) if threshold not in ("", None) else 0.6
        query_embedding = self.dense_model.encode(query)
        results = []
        with self.db.driver.session() as session:
            db_results = session.run(
                "MATCH (e:Entity) RETURN e.dish_mapping_id as dish_mapping_id, e.nome as nome, e.embedding as embedding, e.text as text"
            )
            for record in db_results:
                dish_mapping_id = record.get("dish_mapping_id")
                entity_name = record.get("nome")
                embedding_str = record.get("embedding")
                text = record.get("text")
                if embedding_str:
                    try:
                        entity_embedding = np.array(json.loads(embedding_str))
                    except Exception:
                        continue
                    score = self.compute_similarity(query_embedding, entity_embedding)
                    if score > threshold:
                        results.append({
                            "dish_mapping_id": dish_mapping_id,
                            "name": entity_name,
                            "score": score,
                            "text": text
                        })
        # Sort the results by descending score and return only the top_k
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        return {"results": results}

    @staticmethod
    def compute_similarity(embedding1, embedding2):
        """Calculate the cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(embedding1, embedding2) / (norm1 * norm2)

    def answer_question(self, query: str, threshold: float = 0.6, top_k: int = 5) -> dict:
        """
        Uses the embedding search to extract context texts and then asks an LLM,
        using the extracted texts as context, to determine which entity id best
        answers the user's question.

        Args:
            query (str): The user's question.
            threshold (float): Similarity threshold for filtering entities.
            top_k (int): Maximum number of entities to consider.

        Returns:
            dict: A dictionary containing:
                - "results": The filtered list of matching entities.
                - "llm_answer": The entity ids selected by the LLM.
        """
        # Step 1: Retrieve similar entities via embedding search
        search_results = self.retrieve_entities(query, threshold, top_k)
        entities = search_results.get("results", [])
        
        if not entities:
            return {"results": [], "llm_answer": "0"}

        # Step 2: Combine the texts from the retrieved entities to form the context
        context_texts = "\n".join(
            f'**{entity["name"]} ({entity["dish_mapping_id"]})**\n{entity["text"]}\n' 
            for entity in entities if entity.get("text")
        )

        # Step 3: Build the prompt for the LLM.
        prompt = f"""
    You are an expert in information retrieval.
    Based on the following context texts extracted from our database entities, determine which entity id best answers the user question.
    Only provide the entity ids in your answer.

    User Question:
    {query}

    Context Texts:
    {context_texts}

    Please return ONLY a list of comma-separated ids.
        """

        # Step 4: Query the LLM
        llm_response = self.llm.generate(messages=[{"role": "user", "content": prompt}])
        answer_ids = set(llm_response.strip().split(","))  # Converti in set per il confronto rapido

        # Step 5: Filtra le entità per restituire solo quelle presenti nella risposta dell'LLM
        filtered_entities = [entity for entity in entities if str(entity["dish_mapping_id"]) in answer_ids]

        return {
            "results": filtered_entities,
            "llm_answer": llm_response
        }


    def close(self):
        """Close the connection to the database."""
        self.db.close()
