import json
import re
import os
import numpy as np
from sentence_transformers import SentenceTransformer

from utils import config
from utils import database
from utils.llm import Ollama as LLM

class CypherRAGRetriever:
    def __init__(self):
        self.db = database.Neo4jClient()
        self.model_name = config.MODEL_NAME
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.feedback_file = config.FEEDBACK_FILE
        self.feedback_store = self.load_feedback()
        self.llm = LLM(self.model_name)

    def close(self):
        """Chiude la connessione con Neo4j."""
        self.db.close()

    def get_database_schema(self):
        """
        Recupera la struttura dettagliata del database Neo4j, includendo:
        - Nodi con le loro proprietà usando APOC
        - Relazioni con tipo, nodi di partenza/arrivo, proprietà e statistiche
        """
        schema_data = {
            "nodes": {},
            "relationships": {}
        }

        with self.db.driver.session() as session:
            # Recupera i nodi e le loro proprietà con APOC
            node_query = """
            CALL apoc.meta.data()
            YIELD label, type, other, elementType, property
            WHERE elementType = 'node'
            RETURN label, collect({name: property, type: type}) AS properties
            """
            result = session.run(node_query)
            for record in result:
                node_type = record["label"]
                properties = [
                    {
                        "name": prop["name"].lower(),  # I nomi delle proprietà sono normalizzati
                        "type": prop["type"]
                    } for prop in record.get("properties", [])
                ]
                schema_data["nodes"][node_type] = properties

            # Recupera informazioni dettagliate sulle relazioni
            rel_query = """
            MATCH (start)-[r]->(end)
            WITH type(r) AS relType,
                 startNode(r) AS startNode,
                 endNode(r) AS endNode,
                 r,
                 labels(startNode(r)) AS startLabels,
                 labels(endNode(r)) AS endLabels
            WITH relType,
                 startLabels,
                 endLabels,
                 properties(r) AS relProperties,
                 count(r) AS relationshipCount,
                 collect(DISTINCT {
                     start: properties(startNode),
                     rel: properties(r),
                     end: properties(endNode)
                 })[0..5] AS examples
            RETURN 
                relType,
                startLabels[0] AS startLabel,
                endLabels[0] AS endLabel,
                keys(relProperties) AS propertyKeys,
                relationshipCount,
                examples
            """
            result = session.run(rel_query)
            for record in result:
                rel_type = record["relType"]
                schema_data["relationships"][rel_type] = {
                    "start": record["startLabel"],
                    "end": record["endLabel"],
                    "properties": record["propertyKeys"],
                    "count": record["relationshipCount"],
                    "examples": record["examples"]
                }

        return schema_data

    def generate_schema_context(self, schema_data: dict) -> str:
        """
        Genera una rappresentazione testuale dello schema del database Neo4j secondo il formato:
        - Sezione Nodes con proprietà e datatype (NOTA: le proprietà 'nome' sono normalizzate in minuscolo)
        - Sezione Relationships con proprietà e datatype
        - Sezione Relationships tra nodi (pattern)
        """
        lines = []

        # Sezione Nodes
        lines.append("## Nodes with properties and datatype (tutti i nomi sono normalizzati in minuscolo):")
        nodes = schema_data.get("nodes", {})
        if not nodes:
            lines.append("Nessun nodo trovato.")
        else:
            for node_type, properties in nodes.items():
                if properties:
                    props_str = ",".join(f"{prop['name']}: {prop['type']}" for prop in properties)
                    lines.append(f"(:{node_type} {{ {props_str} }})")
                else:
                    lines.append(f"(:{node_type})")

        # Sezione Relationships with properties and datatype
        lines.append("\n## Relationships with properties and datatype:")
        relationships = schema_data.get("relationships", {})
        rels_with_props = []
        for rel_type, details in relationships.items():
            props = details.get("properties", [])
            if props:
                props_str = ",".join(f"{prop}: UNKNOWN" for prop in props)
                rels_with_props.append(f"[:{rel_type} {{ {props_str} }}]")
        if rels_with_props:
            lines.extend(rels_with_props)
        else:
            lines.append("Nessuna relazione con proprietà trovata.")

        # Sezione Relationships between nodes
        lines.append("\n## Relationships between nodes:")
        patterns = set()
        for rel_type, details in relationships.items():
            start_node = details.get("start")
            end_node = details.get("end")
            if start_node and end_node:
                patterns.add(f"(:{start_node})-[:{rel_type}]->(:{end_node})")
        if patterns:
            for pattern in sorted(patterns):
                lines.append(pattern)
        else:
            lines.append("Nessuna relazione trovata.")

        return "\n".join(lines)

    def get_base_query(self):
        """
        Restituisce la query base con tutte le relazioni possibili.
        I nomi nei nodi sono attesi in forma normalizzata (minuscola).
        """
        return """
        MATCH (p:Piatto)-[:CREATO_DA]->(c:Chef)
        MATCH (p)-[:HA_INGREDIENTE]->(i:Ingrediente)
        MATCH (p)-[:SERVITO_IN]->(r:Ristorante)
        MATCH (p)-[:USA_TECNICA]->(t:Tecnica)
        MATCH (r)-[:SITUATO_IN]->(pn:Pianeta)
        MATCH (r)-[:HA_LICENZA]->(l:Licenza)
        MATCH (c)-[:LAVORA_IN]->(r)
        MATCH (c)-[:HA_CERTIFICAZIONE]->(l)
        MATCH (og:OrdineGalattico)-[:VIETA]->(p)
        MATCH (og)-[:HA_RESTRIZIONE]->(res:Restrizione)
        MATCH (pn)-[:DISTA]->(pn2:Pianeta)
        MATCH (t)-[:RICHIEDE]->(l)
        RETURN p
        """

    def get_feedback_query_for_question(self, user_question):
        """Restituisce una query dal feedback esistente per una domanda specifica, se esiste."""
        user_question = user_question.strip().lower()

        for feedback in self.feedback_store.values():
            stored_question = feedback["user_question"].strip().lower()
            if stored_question == user_question:
                return feedback["cypher_query"]

        return None

    def modify_base_query(self, base_query: str, user_question: str, schema_context: str) -> str:
        """
        Modifica la query base rimuovendo parti inutili e aggiungendo condizioni basate sulla domanda dell'utente.
        """
        prompt = f"""
You are an expert in Neo4j and Cypher.
The following is the base query, which includes all possible relationships and the database schema.

Schema Context:
{schema_context}

Base Query:
{base_query}

Your task is to modify the base query to answer the user question by:
- Identity entities contained in the user query and classify them (Ingrediente, Tecnica, Licenza, Ordine, Pianeta, Chef, Ristorante). 
- In user question entities are always written in capital letters except for conjunctions, prepositions, and articles. (ex. Essenza di Tachioni, Affumicatura a Stratificazione Quantica).
- Convert entities to lower to include them in the cypher query.
- Adding a MATCH clause for each condition specified in the user question.
- Prefer MATCH with condition over WHERE for simple queries. ex MATCH (p)-[:HA_INGREDIENTE]->(i:Ingrediente {{nome: 'ingredient to search'}})
- Never change an Entity name. Use names exactly as provided by user.
- Never MATCH a Piatto by name.
- Remove unnecessary MATCH clauses that are irrelevant to answering the user's question.
- Keeping the RETURN statement unchanged. Always return the dish (p).
- Removing unnecessary MATCH clauses that are irrelevant to answering the user's question.
- Use only relationships in base query. Relationships are always written in capital letters. ex. [:HA_INGREDIENTE]

User Question:
{user_question}
    
"""

        if len(self.feedback_store.values()) > 0:
            prompt += "Use this examples queries for your reference:\n"

            # Aggiungi i feedback precedenti come esempi di few-shot learning
            for feedback in self.feedback_store.values():
                example = f"User Question:\n{feedback['user_question']}\nCypher Query:\n{feedback['cypher_query']}\n"
                prompt += example
                prompt += "\n-----------\n"

        prompt += "Generate ONLY the Cypher query without any explanation or markdown."
                
        response = self.llm.generate(messages=[{"role": "user", "content": prompt}])
        modified_query = self.extract_cypher_query(response)
        return modified_query

    def extract_cypher_query(self, text: str) -> str:
        """
        Estrae la query Cypher dal testo generato, cercando il contenuto nel blocco di codice.
        """
        pattern = r"```cypher\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def generate_cypher_query_plan(self, user_question: str, schema_context: str) -> str:
        """
        Genera un piano in linguaggio naturale (bullet points) che descrive i passaggi
        necessari per rispondere alla domanda, basandosi sullo schema fornito.
        """
        prompt = f"""
You are a data scientist with deep understanding of Neo4j and the Cypher query language.
Based on the following database schema context, generate a detailed plan (in bullet points)
that outlines the steps required to answer the user question. The plan should only describe the logical steps,
including which nodes, relationships, and filters to use.
Remember: the final query must always return the dish (p).

Schema Context:
{schema_context}

User question:
{user_question}

Return ONLY the plan as bullet points.
"""
        response = self.llm.generate(messages=[{"role": "user", "content": prompt}])
        plan_text = response
        return plan_text.strip()

    def generate_cypher_query_from_plan(self, plan: str, schema_context: str) -> str:
        """
        Trasforma il piano in una query Cypher valida.
        """
        prompt = f"""
You are an expert in Neo4j and the Cypher query language.
Based on the following plan (provided in bullet points) and the database schema context,
generate a valid Cypher query that answers the user question.
Make sure to follow the schema exactly, using the correct node labels, relationship types, and directions.
Additionally, ensure that the query always returns both the dish (p).

Plan:
{plan}

Schema Context:
{schema_context}

Return ONLY the Cypher query without explanation or markdown.
"""
        response = self.llm.generate(messages=[{"role": "user", "content": prompt}])
        query_text = response
        cypher_query = self.extract_cypher_query(query_text)
        return cypher_query

    def validate_and_correct_query(self, cypher_query: str, schema_context: str) -> str:
        """
        Valida la query Cypher contro lo schema fornito e la corregge se necessario.
        Invia il prompt a Ollama per verificare la correttezza della query secondo lo schema,
        includendo un controllo per errori di sintassi, come l'introduzione di nuove variabili
        in pattern expressions. Inoltre, verifica che la query ritorni sempre il piatto (p).
        """
        prompt = f"""
You are an expert in Neo4j and the Cypher query language.
Validate the following Cypher query against the provided schema context.
Additionally, check that:
- The query does not introduce new variables in pattern expressions.
- The query always returns the dish (p).
- The query never modifies data.
If the query has any errors regarding schema usage or syntax, correct them and return ONLY the corrected Cypher query without any additional explanation.
If the query is valid, return it unchanged.

Schema Context:
{schema_context}

Query to validate:
{cypher_query}
"""
        response = self.llm.generate(messages=[{"role": "user", "content": prompt}])
        corrected_text = response
        corrected_query = self.extract_cypher_query(corrected_text)
        return corrected_query

    def generate_and_validate_query(self, user_question: str, schema_context: str) -> str:
        """
        Genera una query Cypher a partire dalla query base, la modifica e la valida.
        """
        base_query = self.get_base_query()
        
        # Step 1: Modifica la query base in base alla domanda dell'utente
        modified_query = self.modify_base_query(base_query, user_question, schema_context)
        return modified_query
        
        # Step 2: Valida e corregge la query se necessario
        validated_query = self.validate_and_correct_query(modified_query, schema_context)
        return validated_query

    def execute_query(self, cypher_query: str):
        """
        Esegue una query Cypher personalizzata e restituisce i risultati.
        """        
        try:
            with self.db.driver.session() as session:
                result = session.run(cypher_query)
                records = list(result)
                if not records:
                    return []
                return [dict(record) for record in records]
        except Exception as e:
            error_msg = str(e)
            #print(f"⚠️ Errore nell'esecuzione della query: {error_msg}")
            return [-1]

    def answer_question(self, user_question: str, repeat_on_error: bool = True):
        """
        Risponde alla domanda dell'utente generando ed eseguendo una query Cypher.
        I nomi e le domande sono gestiti in forma normalizzata (minuscola).
        """
        user_question = user_question.strip().lower()
        try:
            
            # Controlla se esiste un feedback valido per la domanda dell'utente
            feedback_query = self.get_feedback_query_for_question(user_question)
            if feedback_query:
                results = self.execute_query(feedback_query)
                return {
                    "results": results,
                    "query": feedback_query
                }
             
            # Recupera lo schema del database
            schema_data = self.get_database_schema()
            schema_context = self.generate_schema_context(schema_data)
            
            # Genera e valida la query combinando i tre step
            validated_query = self.generate_and_validate_query(user_question, schema_context)

            # Esegue la query su Neo4j
            results = self.execute_query(validated_query)
            
            # Se la query restituisce un errore, riprova per una volta a generare la query
            if(results == [-1] and repeat_on_error):
                return self.answer_question(user_question, False)
                
            return {
                "results": results,
                "query": validated_query
            }
        except Exception as e:
            print(f"❌ Errore: {str(e)}")
            return {}
        
    def load_feedback(self):
        """Carica i feedback esistenti dal file JSON."""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                return {}
        return {}

    def find_similar_entities(self, query_text: str, threshold: float = 0.60):
        """
        Trova entità con embedding simili nel database e ritorna anche il testo associato.
        I nomi delle entità sono gestiti in forma normalizzata.
        """
        query_text = query_text.strip().lower()
        query_embedding = self.dense_model.encode(query_text)
        with self.db.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) RETURN e.name as name, e.text as text, e.embedding as embedding"
            )
            similar_entities = []
            for record in result:
                # Confrontiamo i nomi normalizzati
                entity_name = record["name"].strip().lower()
                entity_text = record["text"]
                entity_embedding = np.array(json.loads(record["embedding"]))
                similarity = self.compute_similarity(query_embedding, entity_embedding)
              
                if similarity > threshold:
                    similar_entities.append((entity_name, entity_text, similarity))
            return sorted(similar_entities, key=lambda x: x[2], reverse=True)

    @staticmethod
    def compute_similarity(embedding1, embedding2):
        """Calcola la similarità coseno tra due embedding."""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


if __name__ == "__main__":
    retriever = CypherRAGRetriever()
    # Domanda di esempio
    question = "Quali piatti preparati al ristorante L'Essenza dell'Infinito utilizzano Fibra di Sintetex o Essenza di Vuoto?"
    # Esecuzione della ricerca di entità simili
    similar = retriever.find_similar_entities(question)
    print("Entità simili trovate:")
    for name, text, sim in similar:
        print(f"Nome: {name} | Similarità: {sim:.2f} | Testo: {text}")

    # Esecuzione della risposta alla domanda
    response = retriever.answer_question(question)

    for record in response.get('results', []):
        node = record.get("p")
        if node:
            nome = node.get("nome")
            id_val = node.get("id")
            print("Nome:", nome, "- ID:", id_val)
        else:
            print("Struttura record non prevista:", record)
    
    retriever.close()
