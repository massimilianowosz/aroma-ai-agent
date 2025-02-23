from typing import List, Optional, Dict
from nltk.corpus import stopwords
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchText
from qdrant_client.models import (
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector as ModelSparseVector,
)
from fuzzywuzzy import process
from sentence_transformers import SentenceTransformer  # per embedding dense
import pprint

# Scarica le stopwords italiane se non sono già presenti
# nltk.download("stopwords")
italian_stopwords = set(stopwords.words("italian"))

class KeywordSearcher:
    def __init__(
        
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "menu_collection",
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.keywords = self.load_keywords()

    def load_keywords(self) -> set:
        """
        Recupera ingredienti e tecniche uniche in modo più efficiente usando `scroll()`.
        """
        keywords = set()
        results = self.client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=["ingredienti", "tecniche"],
        )
        for hit in results[0]:
            if "ingredienti" in hit.payload:
                keywords.update(hit.payload["ingredienti"].split(", "))
            if "tecniche" in hit.payload:
                keywords.update(hit.payload["tecniche"].split(", "))
        return keywords

    def extract_keywords_from_query(self, query: str) -> str:
        """
        Identifica le keyword rilevanti nella query utilizzando fuzzy matching.
        """
        query_words = query.lower().split()
        matched_keywords = []
        for word in query_words:
            match, score = process.extractOne(word, self.keywords)
            if score > 85:
                matched_keywords.append(match)
        return " ".join(matched_keywords) if matched_keywords else query

    def clean_text(self, text: str) -> str:
        """
        Rimuove le stopwords dal testo mantenendo la leggibilità.
        """
        return " ".join([word for word in text.split() if word.lower() not in italian_stopwords])

    def keyword_search(
    self, 
    query: List[str],
    top_k: int = 25,
    exclude_query: Optional[List[str]] = None,
    weight_sparse: float = 1.0,
    weight_dense: float = 1.0,
    weight_fulltext: float = 1.0,
    min_score: float = 0.1,
    search_fields: Dict = ["nome_piatto", "ingredienti", "tecniche", "chef", "ristorante", "testo_piatto", "testo_generale"]
) -> Dict:
        """
        Esegue una ricerca combinata su:
        - Sparse search (BM42)
        - Dense vector search
        - Full‑Text search
        Fondendo i risultati in base ai pesi specificati per ogni modalità.

        Args:
            query (List[str]): Lista dei termini di ricerca.
            top_k (int, optional): Numero di risultati da restituire.
            exclude_query (Optional[List[str]]): Termini da escludere.
            weight_sparse (float, optional): Peso per i risultati della ricerca sparse.
            weight_dense (float, optional): Peso per i risultati della ricerca dense.
            weight_fulltext (float, optional): Peso per i risultati della ricerca full‑text.
            min_score (float, optional): Punteggio minimo per includere un risultato.

        Returns:
            Dict: Risultati della ricerca con punteggi fusi.
        """
        # Combina la lista in una query singola ed eventualmente la raffina
        combined_query = " ".join(query)
        refined_query = combined_query

        # --- Ricerca Sparse ---
        sparse_query_obj = list(self.sparse_model.embed([refined_query]))[0]
        prefetch = [
            Prefetch(
                query=ModelSparseVector(
                    indices=sparse_query_obj.indices.tolist(),
                    values=sparse_query_obj.values.tolist()
                ),
                using="bm42",
                limit=top_k,
            )
        ]
        fusion_query = FusionQuery(fusion=Fusion.RRF)
        sparse_results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=fusion_query,
            with_payload=True,
        )

        # --- Ricerca Dense ---
        dense_query_vector = self.dense_model.encode([refined_query])[0]
        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_query_vector,  # Query vettoriale dense
            with_payload=True,
            limit=top_k,
            using="dense"
        )

        # --- Ricerca Full‑Text ---
        positive_filters = []
        for term in query:
            for field in search_fields:
                positive_filters.append(FieldCondition(key=field, match=MatchText(text=term)))

        negative_filters = None
        if exclude_query:
            negative_filters = []
            for term in exclude_query:
                for field in search_fields:
                    negative_filters.append(FieldCondition(key=field, match=MatchText(text=term)))

        final_filter = Filter(
            should=positive_filters,
            must_not=negative_filters
        )
        fulltext_results = self.client.query_points(
            collection_name=self.collection_name,
            query_filter=final_filter,
            with_payload=True,
            limit=top_k,
        )

        # --- Fusione dei risultati ---
        results_dict = {}

        # Aggiunge i risultati della ricerca sparse
        for hit in sparse_results.points:
            results_dict[hit.id] = {
                "id": hit.id,
                "score": weight_sparse * hit.score,
                "text": hit.payload.get("testo_piatto", ""),
                "metadata": hit.payload,
            }
        # Aggiunge (o somma) i risultati della ricerca dense
        for hit in dense_results.points:
            if hit.id in results_dict:
                results_dict[hit.id]["score"] += weight_dense * hit.score
            else:
                results_dict[hit.id] = {
                    "id": hit.id,
                    "score": weight_dense * hit.score,
                    "text": hit.payload.get("testo_piatto", ""),
                    "metadata": hit.payload,
                }
                
        # Aggiunge (o somma) i risultati della ricerca full‑text
        for hit in fulltext_results.points:
            if hit.id in results_dict:
                results_dict[hit.id]["score"] += weight_fulltext * hit.score
            else:
                results_dict[hit.id] = {
                    "id": hit.id,
                    "score": weight_fulltext * hit.score,
                    "text": hit.payload.get("testo_piatto", ""),
                    "metadata": hit.payload,
                }

        merged_results = sorted(results_dict.values(), key=lambda x: x["score"], reverse=True)
                
        # Filtro dei risultati basato sul min_score
        filtered_results = [result for result in merged_results if result["score"] >= min_score]

        return {
            "query": query,
            "exclude_query": exclude_query,
            "results": filtered_results[:top_k]
        }


if __name__ == '__main__':
    # Istanzia il KeywordSearcher con i parametri predefiniti
    searcher = KeywordSearcher(
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="menu_collection"
    )

    # Definisci la query di test (ad esempio, una lista con un termine)
    test_query = ["Latte+"]

    # Esegui la ricerca con top_k impostato a 10
    results = searcher.keyword_search(
        query=test_query,
        top_k=10,
        exclude_query=None,
        weight_sparse=0.7,
        weight_dense=0.4,
        weight_fulltext=1
    )

    # Stampa i risultati in maniera formattata
    pprint.pprint(results)