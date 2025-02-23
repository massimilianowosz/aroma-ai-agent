# AROMA: Artificial Retrieval and Optimization for Multiversal Appetites

## 🚀 Overview

**AROMA** (Augmented Retrieval-Oriented Multiverse Assistant) è un sistema di retrieval ibrido basato su **Neo4j** e modelli di **AI generativa**, progettato per supportare la navigazione culinaria intergalattica. L'agente utilizza un approccio avanzato di **Chain of Thought (CoT)** per selezionare dinamicamente le strategie di retrieval più efficaci e restituire risultati precisi e contestualizzati.

---

## 🔧 Architettura

AROMA combina retrieval strutturato e semantico per massimizzare **precisione** e **recall**:

- **Cypher Search Tool** → Query strutturate su **Neo4j** per interrogare relazioni esplicite nel grafo.
- **Vector Search Tool** → Ricerca basata su **embedding vettoriali** per individuare concetti affini.
- **Adaptive Few-Shot Prompting** → Ottimizzazione dinamica delle query basata su feedback utente.
- **Prompt Inject Guard** → Protezione contro **prompt injection** e input non validi.
- **Structured ID Validation** → Controllo incrociato degli ID con la knowledge base.

---

## 🏗️ Pipeline di Ingestion

Il sistema elabora i dati attraverso una pipeline di ingestion composta da:

1. **Parsing e Normalizzazione** → Estrazione delle entità rilevanti da PDF e database.
2. **Embedding & Indicizzazione** → Creazione di vettori semantici e modellazione in **Neo4j**.
3. **Relazioni & Validazione** → Costruzione del knowledge graph e pulizia dei dati.

---

## 🔍 Flusso Chain of Thought

L’agente segue un processo iterativo **Thought → Action → Observation** per ottimizzare la selezione delle strategie di retrieval:

1. **Thought** → Analizza la richiesta, identifica le entità e pianifica il retrieval.
2. **Action** → Seleziona ed esegue la query più appropriata (**Cypher** o vettoriale).
3. **Observation** → Se i risultati sono scarsi o assenti, avvia una fallback strategy:
   - **HITL Feedback** → Chiede input all'utente per migliorare la query.
   - **Adaptive Few-Shot Prompting** → Usa esempi precedenti per affinare automaticamente la richiesta.

---

## 🔒 Sicurezza & Controllo

Per garantire **affidabilità e protezione**, AROMA include:

- **Prompt Inject Guard** → Filtro automatico degli input per prevenire exploit.
- **Query Validation** → Controllo sintattico e semantico delle query prima dell’esecuzione.
- **Structured ID Validation** → Verifica la coerenza degli ID restituiti.

---

## 🛠️ Setup & Installazione

### 🔹 Requisiti

- **Python** 3.9+
- **Neo4j** 5+
- **Ollama** per LLM inference
- **Dipendenze Python** (installabili via pip)

### 🔹 Installazione

```bash
# Clona il repository
git clone https://github.com/massimilianowosz/aroma-ai-agent
cd aroma

# Installa le dipendenze
pip install -r requirements.txt

# Avvia il sistema
docker compose up -d
```

---

## 📌 Utilizzo

L’agente può essere interrogato direttamente o tramite **process.py**:

```bash
python -m agent
python process.py
```

---

## 📜 Licenza

Questo progetto è rilasciato sotto licenza **MIT**.

---

## 📫 Contatti

Per qualsiasi informazione o contributo, **apri una issue su GitHub**.