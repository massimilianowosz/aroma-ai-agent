import fitz 
import json
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import emoji
import uuid

from utils import config
from utils import database
from utils.llm import Ollama as LLM

class RAGIndexer:
    def __init__(self, clear_database: bool = False):
        self.db = database.Neo4jClient()
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model_name = config.MODEL_NAME
        self.llm = LLM(self.model_name)

        with open(config.DATASET_PATH+"/Misc/dish_mapping.json", "r", encoding="utf-8") as f:
            # Assicuriamoci che anche le chiavi della dish mapping siano in minuscolo
            self.dish_mapping = {k.lower(): v for k, v in json.load(f).items()}
        
        if clear_database:
            self.clear_database()
            
    def process_planets(self):
        self.planets = self.load_planets(config.DATASET_PATH+"/Misc/Distanze.csv")
        self.insert_planets(config.DATASET_PATH+"/Misc/Distanze.csv")
        self.insert_planet_distances(config.DATASET_PATH+"/Misc/Distanze.csv")
        
    def close(self):
        self.db.close()
    
    def load_planets(self, file_path):
        df = pd.read_csv(file_path, index_col=0)
        planet_mapping = {}
        for planet in df.columns:
            planet_lower = planet.lower()
            planet_mapping[planet_lower] = planet_lower
        return planet_mapping
    
    def insert_planets(self, file_path):
        df = pd.read_csv(file_path, index_col=0)
        with self.db.driver.session() as session:
            for planet in df.index:
                # converto in minuscolo
                session.execute_write(self._create_planet_node, planet.lower())
    
    @staticmethod
    def _create_planet_node(tx, planet):
        planet_uuid = str(uuid.uuid4())
        query = "MERGE (p:Pianeta {nome: $planet}) ON CREATE SET p.uuid = $planet_uuid"
        tx.run(query, planet=planet, planet_uuid=planet_uuid)
    
    def insert_planet_distances(self, file_path):
        df = pd.read_csv(file_path, index_col=0)
        with self.db.driver.session() as session:
            for planet_a in df.index:
                for planet_b in df.columns:
                    if planet_a != planet_b:
                        distanza = df.loc[planet_a, planet_b]
                        # Passo i nomi in minuscolo
                        session.execute_write(self._create_distance_relation, planet_a.lower(), planet_b.lower(), distanza)
    
    @staticmethod
    def _create_distance_relation(tx, planet_a, planet_b, distanza):
        query = (
            "MATCH (a:Pianeta {nome: $planet_a}) "
            "MATCH (b:Pianeta {nome: $planet_b}) "
            "MERGE (a)-[:DISTA {anni_luce: $distanza}]->(b)"
        )
        tx.run(query, planet_a=planet_a, planet_b=planet_b, distanza=distanza)
    
    def extract_planet_from_text(self, general_text):
        """Trova il pianeta menzionato nel testo generale."""
        for planet in self.planets:
            if re.search(rf'\b{re.escape(planet)}\b', general_text, re.IGNORECASE):
                return planet
        return "sconosciuto"

    def extract_licenses(self, general_text):
        """Estrae le licenze degli chef e ristoranti con attributi nome e grado."""
        
        prompt = (
            "Nel seguente testo vengono menzionate diverse licenze di cucina che uno chef o ristorante può possedere.\n"
            "Di seguito le licenze valide:\n"
            """
            - Psionica (acronimo: P)
            - Temporale (acronimo: T)
            - Gravitazionale (acronimo: G)
            - Antimateria (acronimo: e+)
            - Magnetica (acronimo: Mx)
            - Quantistica (acronimo: Q)
            - Luce (acronimo: c)
            - Livello di Sviluppo Tecnologico (acronimo: LTK)
            """
            "Estrai SOLO i nomi delle licenze presenti nel testo, e suddividi il nome, l'acronimo e il grado.\n"
            "Converti il grado in un numero intero se espresso in numeri romani o a parole (es. primo, secondo, terzo ecc).\n"
            "Restituisci la risposta in formato JSON con la chiave 'licenze', dove ogni licenza ha attributi 'nome', 'acronimo' e 'grado' come numero.\n\n"
            f"Testo:\n{general_text}\n\n"
            "Esempio di risposta JSON valida:\n"
            '{ "licenze": [{"nome": "psionica", "acronimo": "P", "grado": 1}, {"nome": "antimateria", acronimo:"e+", "grado": 2}] }'
        )
        
        response = self.llm.generate(
            format="json",
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            content = json.loads(response)
            licenze_estratte = content.get("licenze", [])
            
            return licenze_estratte if licenze_estratte else [{"nome": "non specificato", "acronimo": "non specificato", "grado": "non specificato"}]
        except json.JSONDecodeError:
            return [{"nome": "non specificato", "acronimo": "non specificato", "grado": "non specificato"}]

    def insert_piatto(self, dish, extracted_data, ristorante, pianeta):
        # Converto il nome del piatto in minuscolo
        dish_lower = dish.lower()
        # Recupera l'ID del piatto dal JSON, se non esiste genera un UUID
        dish_id = self.dish_mapping.get(dish_lower, None)
        
        if dish_id is None:
            print("Dish id non trovato")
            exit()
        
        with self.db.driver.session() as session:
            session.execute_write(self._create_piatto, dish_lower, dish_id, extracted_data, ristorante.lower(), pianeta.lower())

    @staticmethod
    def _create_piatto(tx, dish, dish_id, extracted_data, ristorante, pianeta):
        dish_uuid = str(uuid.uuid4())
        # Assicuriamoci che il nome dello chef sia in minuscolo
        chef = extracted_data['chef'].lower() if extracted_data.get('chef') else "non specificato"
        query = (
            "MERGE (p:Piatto {nome: $nome_piatto}) "
            "ON CREATE SET p.uuid = $dish_uuid, p.dish_mapping_id = $dish_id "
            "MERGE (r:Ristorante {nome: $ristorante}) "
            "MERGE (r)-[:SITUATO_IN]->(pl:Pianeta {nome: $pianeta}) "
            "MERGE (p)-[:SERVITO_IN]->(r) "
            "FOREACH (ing IN $ingredienti | "
            "MERGE (i:Ingrediente {nome: ing}) ON CREATE SET i.uuid = randomUUID() "
            "MERGE (p)-[:HA_INGREDIENTE]->(i)) "
            "FOREACH (tec IN $tecniche | "
            "MERGE (t:Tecnica {nome: tec}) ON CREATE SET t.uuid = randomUUID() "
            "MERGE (p)-[:USA_TECNICA]->(t)) "
            "MERGE (c:Chef {nome: $chef}) ON CREATE SET c.uuid = $chef_uuid "
            "MERGE (c)-[:LAVORA_IN]->(r) "
            "MERGE (p)-[:CREATO_DA]->(c) "
        )
        chef_uuid = str(uuid.uuid4())
        # I valori ingredienti e tecniche li separiamo e non serve ulteriormente la conversione qui
        tx.run(query,
            dish_id=dish_id,
            nome_piatto=dish,
            dish_uuid=dish_uuid,
            ingredienti=extracted_data['ingredienti'].split(", "),
            tecniche=extracted_data['tecniche'].split(", "),
            chef=chef,
            ristorante=ristorante,
            pianeta=pianeta,
            chef_uuid=chef_uuid
        )

    def insert_ristorante(self, nome, pianeta, chef, licenze):
        """Crea un nodo per il ristorante con informazioni generali."""
        with self.db.driver.session() as session:
            session.execute_write(self._create_ristorante, nome.lower(), pianeta.lower(), chef.lower(), licenze)

    @staticmethod
    def _create_ristorante(tx, nome, pianeta, chef, licenze):
        ristorante_uuid = str(uuid.uuid4())
        chef_uuid = str(uuid.uuid4())
        planet_uuid = str(uuid.uuid4())
        for licenza in licenze:
            if not licenza.get('grado'):
                licenza['grado'] = 0

        query = (
            "MERGE (r:Ristorante {nome: $nome}) ON CREATE SET r.uuid = $ristorante_uuid "
            "MERGE (pl:Pianeta {nome: $pianeta}) ON CREATE SET pl.uuid = $planet_uuid "
            "MERGE (r)-[:SITUATO_IN]->(pl) "
            "MERGE (c:Chef {nome: $chef}) ON CREATE SET c.uuid = $chef_uuid "
            "MERGE (c)-[:LAVORA_IN]->(r) "
            "FOREACH (lic IN $licenze | "
            "MERGE (l:Licenza {nome: toLower(lic.nome), acronimo: toLower(lic.acronimo), grado: lic.grado})"
            "MERGE (c)-[:HA_CERTIFICAZIONE]->(l) "
            "MERGE (r)-[:HA_LICENZA]->(l))"
        )
        tx.run(query, nome=nome, ristorante_uuid=ristorante_uuid, pianeta=pianeta, chef=chef,
               licenze=licenze, chef_uuid=chef_uuid, planet_uuid=planet_uuid)

    # -------------------------
    # Modifica del metodo process_pdf per inserire gli embedding
    # -------------------------
    def process_pdf(self, pdf_path: Path):
        print(f"\n📄 Analizzando {pdf_path.name}...")
        extracted_data = self.extract_text_with_metadata(pdf_path)
        sections = self.extract_sections_from_text(extracted_data)
        
        # Il testo generale descrive il ristorante, non un piatto
        general_text = " ".join(sections.pop("general_text", []))
        
        # Estrae ristorante, chef e pianeta
        global_ristorante, global_chef = self.extract_chef_from_intro(general_text)
        # Converto in minuscolo
        global_ristorante = global_ristorante.lower() if global_ristorante else "non specificato"
        global_chef = global_chef.lower() if global_chef else "non specificato"
        pianeta = self.extract_planet_from_text(general_text)
        
        # Estrarre licenze **del ristorante**
        licenze = self.extract_licenses(general_text)

        # Creiamo il nodo del ristorante (senza associarlo a un piatto)
        self.insert_ristorante(global_ristorante, pianeta, global_chef, licenze)

        # Inserisce gli embedding per Ristorante, Chef e Pianeta
        #self.insert_embedding("ristorante_" + global_ristorante, general_text)
        #self.insert_embedding("chef_" + global_chef, global_chef)
        #self.insert_embedding("pianeta_" + pianeta, pianeta)
        
        # Processa ciascun piatto e inserisce il relativo embedding
        for dish, section in sections.items():
            if not section:
                continue
            dish_text = " ".join(section)
            dish_lower = dish.lower()
            dish_id = self.dish_mapping.get(dish_lower, None)

            self.insert_embedding(dish_lower, dish_text, dish_id)
            
            rules_data = self.extract_by_rules(section)
            llm_data = self.extract_with_llm(section)
            extracted_data_section = self.merge_extraction_results(rules_data, llm_data, global_chef)
            
            self.insert_piatto(dish, extracted_data_section, global_ristorante, pianeta)

    
    def extract_text_with_metadata(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        Estrae il testo da un PDF e include metadati come dimensione del font e stile.

        :param pdf_path: Percorso del file PDF
        :return: Lista di dizionari con testo e metadati
        """
        doc = fitz.open(pdf_path)
        extracted_data = []

        for page in doc:
            blocks = page.get_text("dict")["blocks"]  # Estrai i blocchi di testo con metadati

            for block in blocks:
                if "lines" in block:  # Controlla se il blocco ha linee di testo
                    for line in block["lines"]:
                        for span in line["spans"]:  # Ogni span ha il testo e le informazioni di stile
                            extracted_data.append({
                                "text": span["text"].strip(),
                                "font": span["font"],
                                "size": span["size"],  # Dimensione del carattere
                                "bold": "Bold" in span["font"],  # Controlla se è grassetto
                                "italic": "Italic" in span["font"],  # Controlla se è corsivo
                                "y": block["bbox"][1]  # Posizione verticale nel documento
                            })

        return extracted_data
    
    def extract_sections_from_text(self, extracted_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Identifica i titoli dei piatti e suddivide il testo in sezioni, accodando correttamente ingredienti e descrizioni.

        :param extracted_data: Lista di dizionari contenenti il testo e i metadati (es. font_size, bold, etc.).
        :return: Dizionario con i titoli come chiavi e le sezioni di testo associate, incluso il testo generale.
        """
        sections = {"general_text": []}  # Inizializziamo una sezione per il testo generale
        current_title = None

        for i, item in enumerate(extracted_data):
            line_text = emoji.replace_emoji(item["text"],'').strip()
            font_size = item.get("size", 0)
            is_bold = item.get("bold", False)
            is_italic = item.get("italic", False)  # Se utile in futuro

            if (font_size > 10 or is_bold) and len(line_text) > 3:
                # Confronta in minuscolo
                if line_text.strip().lower() in self.dish_mapping and current_title != line_text:
                    current_title = line_text
                    sections[current_title] = []
                else:
                    if current_title:
                        sections[current_title].append(line_text)
                    else:
                        sections["general_text"].append(line_text)
            elif current_title:
                if sections[current_title] and line_text.startswith(","):
                    sections[current_title][-1] += " " + line_text  # Concatenazione con lo spazio
                else:
                    sections[current_title].append(line_text)
            else:
                sections["general_text"].append(line_text)

        return sections
    
    def extract_chef_from_intro(self, text: str) -> str:
        prompt = (
            "Estrarre il nome del ristorante e dello chef dal seguente testo: \n"
            f"{text}\n\n"
            "Restituisci solo il nome del ristorante e il nome completo dello chef, senza testo aggiuntivo."
            "Fornisci il risultato in formato JSON con le chiavi 'ristorante' e 'nome_chef'.\n"

        )
        
        response = self.llm.generate(
            format="json",
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            content = json.loads(response)
            restaurant_name = content.get("ristorante", "non specificato").strip().lower()
            chef_name = content.get("nome_chef", "non specificato").strip().lower()
            return restaurant_name, chef_name
        except json.JSONDecodeError:
            return "non specificato", "non specificato"
    
    def extract_with_llm(self, section_data: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Usa un LLM per estrarre ingredienti e tecniche dal testo.
        Viene creata una versione "pulita" del testo senza formattazioni per evitare ambiguità.
        NOTA: L'estrazione dello chef non avviene a livello di sezione,
        poiché verrà invece presa dal testo introduttivo.
        """
        cleaned_text = " ".join(section_data)
        
        prompt = (
            "Ti fornisco il testo di un piatto di un ristorante immaginario. "
            "Il tuo compito è estrarre SOLO gli ingredienti e le tecniche presenti, "
            "separandoli con una virgola.\n"
            "Ingredienti e Tecniche da estrarre nel testo fornito iniziano sempre con una lettera maiuscola.\n"
            "Verifica che gli ingredienti anche se immaginari siano riconducibili ad un alimento (una cosa)\n"
            "Verifica che una tecnica anche se immaginaria sia riconducibile ad una tecnica di preparazione (un'azione).\n"
            "Fornisci il risultato in formato JSON con le chiavi 'ingredienti' e 'tecniche'.\n"
            "Ricorda! Non inventare il task è estratte substring dal testo.\n\n"
            "Non includere il nome dello chef in ingredienti o tecniche.\n\n"
            "Testo del piatto:\n" +
            cleaned_text
        )

        response = self.llm.generate(
            format="json",
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            content = json.loads(response)
            return {
                "ingredienti": content.get("ingredienti", "non specificato"),
                "tecniche": content.get("tecniche", "non specificato")
            }
        except json.JSONDecodeError:
            return {"ingredienti": "non specificato", "tecniche": "non specificato"}


    def extract_by_rules(self, section_data: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """
        Estrae ingredienti e tecniche usando regole e regex sul testo formattato.
        Se viene trovata la parola chiave (ad esempio "ingredienti" o "tecniche"),
        vengono processati i token successivi; altrimenti, si usa un fallback basato su regex.
        """
        def extract_category(category: str, tokens: List[Dict[str, str]]) -> List[str]:
            category_lower = category.lower()
            stop_keyword = "tecniche" if category_lower == "ingredienti" else "ingredienti"
            start_index = None

            # Cerca l'inizio della categoria
            for i, token in enumerate(tokens):
                if isinstance(token, str):
                    text_clean = token.strip().lower().replace(":", "")
                elif isinstance(token, dict) and "text" in token:
                    text_clean = token["text"].strip().lower().replace(":", "")
                else:
                    continue  # Ignora token non validi

                if text_clean == category_lower:
                    start_index = i
                    break

            if start_index is not None:
                # Raccoglie i token successivi fino al token di stop
                region_tokens = []
                for token in tokens[start_index+1:]:
                    if isinstance(token, str):
                        token_clean = token.strip().lower().replace(":", "")
                    elif isinstance(token, dict) and "text" in token:
                        token_clean = token["text"].strip().lower().replace(":", "")
                    else:
                        continue  # Ignora token non validi

                    if token_clean == stop_keyword:
                        break
                    region_tokens.append(token)

                # Controlla se ci sono elementi formattati
                has_formatted = any(
                    isinstance(t, dict) and "style" in t and t["style"] in ["bold", "italic"]
                    for t in region_tokens
                )

                if has_formatted:
                    extracted = [
                        t["text"].strip().lower() for t in region_tokens
                        if isinstance(t, dict) and "style" in t and t["style"] in ["bold", "italic"] and t["text"].strip()
                    ]
                else:
                    combined_text = " ".join(
                        t["text"] for t in region_tokens if isinstance(t, dict) and "text" in t
                    )
                    if "\n" in combined_text:
                        extracted = [line.strip().lower() for line in combined_text.split("\n") if line.strip()]
                    elif len(region_tokens) > 1:
                        extracted = [t["text"].strip().lower() for t in region_tokens if isinstance(t, dict) and "text" in t and t["text"].strip()]
                    elif region_tokens:
                        token_text = region_tokens[0]["text"].strip().lower() if isinstance(region_tokens[0], dict) and "text" in region_tokens[0] else ""
                        if "\n" in token_text:
                            extracted = [line.strip().lower() for line in token_text.split("\n") if line.strip()]
                        else:
                            extracted = re.findall(r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){1,3})\b', token_text)
                            extracted = [e.lower() for e in extracted]
                    else:
                        extracted = []
                return extracted
            else:
                # Fallback: applica una regex all'intero testo della sezione
                combined_text = " ".join(
                    token["text"] for token in tokens if isinstance(token, dict) and "text" in token
                )
                candidates = re.findall(r'\b(?:[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})\b', combined_text)
                return [c.strip().lower() for c in candidates if c.strip()]

        ingredienti = extract_category("ingredienti", section_data)
        tecniche = extract_category("tecniche", section_data)
        return {"ingredienti": ingredienti, "tecniche": tecniche}


    def merge_extraction_results(self, rules_data: Dict[str, List[str]], llm_data: Dict[str, object], global_chef: str) -> Dict[str, str]:
        """
        Combina i risultati ottenuti con l'estrazione basata su regole e quella via LLM.
        Rimuove duplicati per ingredienti e tecniche e, per il nome dello chef,
        utilizza il valore estratto dal testo introduttivo (global_chef).
        """
        def merge_category(rule_list, llm_value):
            llm_items = []
            if isinstance(llm_value, list):
                llm_items = [str(item).strip().lower() for item in llm_value if str(item).strip()]
            elif isinstance(llm_value, str) and llm_value.lower() != "non specificato":
                llm_items = [item.strip().lower() for item in llm_value.split(",") if item.strip()]
            merged = list(set(rule_list + llm_items))
            return ", ".join(merged) if merged else "non specificato"


        merged_ingredienti = merge_category(rules_data.get("ingredienti", []), llm_data.get("ingredienti", "non specificato"))
        merged_tecniche = merge_category(rules_data.get("tecniche", []), llm_data.get("tecniche", "non specificato"))
        merged_chef = global_chef if global_chef and global_chef.strip() else "non specificato"
        return {"ingredienti": merged_ingredienti, "tecniche": merged_tecniche, "chef": merged_chef}
        
    def process_menu(self, dataset_path: Path):
        """Processa tutti i PDF nella cartella."""
        pdf_files = list(dataset_path.rglob("*.pdf"))
        if not pdf_files:
            print("⚠️ Nessun PDF trovato nella cartella.")
            return
        for pdf_file in tqdm(pdf_files, desc="📄 Analisi PDF"):
            self.process_pdf(pdf_file)

    def clear_database(self):
        """Cancella tutti i nodi e le relazioni nel database."""
        with self.db.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("🗑️ Database svuotato con successo!")


    def process_codice_galattico(self):
        """ Estrae le licenze tecniche dal Codice Galattico e le inserisce in Neo4j con attributi nome e grado. """
        pdf_path = "dataset/Codice Galattico/Codice Galattico.pdf"
        doc = fitz.open(pdf_path)

        results = {}

        for page_num, page in enumerate(tqdm(doc, desc="📖 Elaborazione del Codice Galattico", unit="pagina")):
            page_text = page.get_text("text").strip()
            if not page_text:
                continue  

            prompt = (
                "Dal seguente testo estrai le informazioni sulle licenze richieste per le tecniche di preparazione. "
                "Di seguito le licenze valide:\n"
                """
                - Psionica (acronimo: P)
                - Temporale (acronimo: T)
                - Gravitazionale (acronimo: G)
                - Antimateria (acronimo: e+)
                - Magnetica (acronimo: Mx)
                - Quantistica (acronimo: Q)
                - Luce (acronimo: c)
                - Livello di Sviluppo Tecnologico (acronimo: LTK)
                """
                "Estrai SOLO i nomi delle licenze presenti nel testo, e suddividi il nome, l'acronimo e il grado.\n"
                "Restituisci i dati in formato JSON con la chiave:\n"
                "'licenze_tecnica': { 'Tecnica': [{'nome': 'LicenzaX', 'acronimo':'X', 'grado': 1}, {'nome': 'LicenzaY', acronimo: 'Y', 'grado': 2}] }\n"
                "Converti il grado in un numero intero se espresso in numeri romani o a parole (es. primo, secondo, terzo ecc).\n"
                "Esempio:\n"
                "{\n"
                "  'licenze_tecnica': { 'Marinatura a infusione gravitazionale': [{'nome': 'Gravitazionale', Acronimo: 'G', 'grado': 2}] }\n"
                "}\n"
                "Non aggiungere altro testo.\n\n"
                f"Testo della pagina {page_num + 1}:\n{page_text}"
            )
            
            response = self.llm.generate(
                format="json",
                messages=[{"role": "user", "content": prompt}]
            )

            try:
                data = json.loads(response)

                for tecnica, licenze in data.get("licenze_tecnica", {}).items():
                    tecnica_lower = tecnica.lower()
                    
                    #Dirty fix per errore di parsing
                    tecnica_lower = tecnica_lower.replace('a(','aff')
                    
                    if tecnica_lower in results:
                        results[tecnica_lower].extend([{"nome": l["nome"].lower(), "acronimo": l["acronimo"].lower(), "grado": l["grado"]} for l in licenze])
                        results[tecnica_lower] = [dict(l) for l in {tuple(l.items()) for l in results[tecnica_lower]}]
                    else:
                        results[tecnica_lower] = [{"nome": l["nome"].lower(), "acronimo":l["acronimo"].lower(),"grado": l["grado"]} for l in licenze]

            except json.JSONDecodeError:
                print(f"Errore nel parsing JSON della pagina {page_num + 1}, passando oltre.")
        
        with self.db.driver.session() as session:
            for tecnica, licenze in results.items():
                for licenza in licenze:
                    if not licenza.get('grado'):
                        licenza['grado'] = 0

                    session.run(
                        """
                        MERGE (t:Tecnica {nome: $tecnica})
                        MERGE (l:Licenza {nome: $nome, acronimo: $acronimo, grado: $grado})
                        MERGE (t)-[:RICHIEDE]->(l)
                        """,
                        {"tecnica": tecnica, "nome": licenza["nome"], "acronimo": licenza["acronimo"], "grado": licenza["grado"]}
                    )

        print("✅ Codice galattico elaborato con successo!")

    def process_manuale_cucina(self):
        """ Estrae gli Ordini Galattici e le loro restrizioni dal Manuale di Cucina e verifica i piatti in base alle restrizioni """
        pdf_path = "dataset/Misc/Manuale di Cucina.pdf"
        doc = fitz.open(pdf_path)

        results = {}

        # Usa tqdm per mostrare l'avanzamento mentre si processano le pagine
        for page_num, page in enumerate(tqdm(doc, desc="📖 Elaborazione del Manuale di Cucina", unit="pagina")):
            page_text = page.get_text("text").strip()
            if not page_text:
                continue  # Salta le pagine vuote

            prompt = (
                "Dal seguente testo estrai le informazioni sugli Ordini Galattici e le loro restrizioni alimentari. "
                "Restituisci i dati in formato JSON con la seguente struttura:\n"
                "{\n"
                "  'ordini_galattici': {\n"
                "    'Ordine della Galassia di Andromeda': ['Restrizione1', 'Restrizione2'],\n"
                "    'Ordine dei Naturalisti': ['Restrizione1', 'Restrizione2'],\n"
                "    'Ordine degli Armonisti': ['Restrizione1', 'Restrizione2']\n"
                "  }\n"
                "}\n"
                "Non aggiungere altro testo.\n\n"
                f"Testo della pagina {page_num + 1}:\n{page_text}"
            )

            response = self.llm.generate(
                format="json",
                messages=[{"role": "user", "content": prompt}]
            )

            try:
                data = json.loads(response)

                # Aggiorna il dizionario con gli Ordini e le loro restrizioni
                for ordine, restrizioni in data.get("ordini_galattici", {}).items():
                    ordine_lower = ordine.lower()
                    if ordine_lower in results:
                        results[ordine_lower].extend([r.lower() for r in restrizioni])
                        results[ordine_lower] = list(set(results[ordine_lower]))  # Evita duplicati
                    else:
                        results[ordine_lower] = [r.lower() for r in restrizioni]

            except json.JSONDecodeError:
                print(f"❌ Errore nel parsing JSON della pagina {page_num + 1}, passando oltre.")

        # Ottieni tutti i piatti con ingredienti e tecniche dal database Neo4j
        piatti = self.get_piatti_from_neo4j()

        # Analizza i piatti rispetto alle restrizioni usando l'LLM
        piatti_vietati = self.verifica_restrizioni_piatti_con_llm(results, piatti)

        # Inserisce i dati in Neo4j
        self.insert_into_neo4j(results, piatti_vietati)

    def get_piatti_from_neo4j(self):
        """ Ottiene tutti i piatti con ingredienti e tecniche dal database Neo4j """
        with self.db.driver.session() as session:
            query = """
            MATCH (p:Piatto)
            OPTIONAL MATCH (p)-[:HA_INGREDIENTE]->(i:Ingrediente)
            OPTIONAL MATCH (p)-[:USA_TECNICA]->(t:Tecnica)
            RETURN p.nome AS Piatto, collect(DISTINCT i.nome) AS Ingredienti, collect(DISTINCT t.nome) AS Tecniche
            """
            result = session.run(query)
            return [
                {
                    "nome": record["Piatto"],
                    "ingredienti": record["Ingredienti"],
                    "tecniche": record["Tecniche"]
                }
                for record in result
            ]

    def verifica_restrizioni_piatti_con_llm(self, results, piatti):
        """ Usa il LLM per verificare quali piatti sono vietati in base alle restrizioni degli Ordini Galattici """
        piatti_vietati = {}

        for ordine, restrizioni in results.items():
            for piatto in tqdm(piatti, desc=f"🔍 Analisi piatti per {ordine}", unit="piatto"):
                prompt = (
                    f"Un ordine galattico ha le seguenti restrizioni alimentari:\n{restrizioni}\n\n"
                    f"Il piatto '{piatto['nome']}' contiene i seguenti ingredienti: {piatto['ingredienti']}.\n"
                    f"Il piatto è preparato con le seguenti tecniche: {piatto['tecniche']}.\n\n"
                    "Questo piatto viola le restrizioni dell'ordine galattico?\n"
                    "Rispondi solo con JSON:\n"
                    "{ 'vietato': true } oppure { 'vietato': false }"
                )

                response = self.llm.generate(
                    format="json",
                    messages=[{"role": "user", "content": prompt}]
                )

                try:
                    data = json.loads(response)
                    if data.get("vietato", False):  # Se vietato è True, lo aggiungiamo
                        if ordine not in piatti_vietati:
                            piatti_vietati[ordine] = []
                            piatti_vietati[ordine].append(piatto["nome"])
                except json.JSONDecodeError:
                    print(f"❌ Errore nell'analisi del piatto {piatto['nome']}, passando oltre.")

        return piatti_vietati

    def insert_into_neo4j(self, results, piatti_vietati):
        """ Inserisce gli Ordini Galattici, le restrizioni e i collegamenti ai piatti vietati in Neo4j """
        with self.db.driver.session() as session:
            for ordine, restrizioni in results.items():
                session.run("MERGE (o:OrdineGalattico {nome: $ordine})", ordine=ordine.lower())

                for restrizione in restrizioni:
                    session.run(
                        """
                        MERGE (r:Restrizione {descrizione: $restrizione})
                        MERGE (o:OrdineGalattico {nome: $ordine})
                        MERGE (o)-[:HA_RESTRIZIONE]->(r)
                        """,
                        {"ordine": ordine.lower(), "restrizione": restrizione.lower()}
                    )

            # Aggiunge le relazioni con i piatti vietati
            for ordine, piatti in piatti_vietati.items():
                for piatto in piatti:
                    session.run(
                        """
                        MATCH (o:OrdineGalattico {nome: $ordine})
                        MATCH (p:Piatto {nome: $piatto})
                        MERGE (o)-[:VIETA]->(p)
                        """,
                        {"ordine": ordine.lower(), "piatto": piatto.lower()}
                    )

        print("✅ Ordini Galattici, restrizioni e piatti vietati inseriti con successo in Neo4j!")

    def normalize_similar_entities(self):
        """
        Confronta e normalizza entità simili (ingredienti, tecniche, licenze) rilevando varianti
        dello stesso concetto scritte in modo diverso.
        """
        print("🔄 Avvio della normalizzazione delle entità...")
        self._normalize_category('Ingrediente')
        self._normalize_category('Tecnica')
        self._normalize_category('Licenza')
        print("✅ Normalizzazione completata!")

    def _normalize_category(self, category_label):
        """
        Normalizza una specifica categoria di entità (Ingrediente, Tecnica, Licenza)
        identificando e unificando varianti simili.
        
        :param category_label: L'etichetta della categoria da normalizzare (es. 'Ingrediente', 'Tecnica', 'Licenza')
        """
        print(f"  ↪ Normalizzazione {category_label}...")
        
        # Ottieni tutte le entità della categoria
        with self.db.driver.session() as session:
            result = session.run(f"MATCH (n:{category_label}) RETURN n.nome as nome")
            entities = [record["nome"] for record in result]
        
        if not entities:
            print(f"  ⚠️ Nessuna entità {category_label} trovata nel database")
            return
        
        # Prima fase: gestione delle varianti che differiscono solo per capitalizzazione
        lowercase_to_variants = {}
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower not in lowercase_to_variants:
                lowercase_to_variants[entity_lower] = []
            lowercase_to_variants[entity_lower].append(entity)
        
        # Gestiamo prima i duplicati esatti (ignorando case)
        exact_matches_processed = set()
        for lowercase, variants in lowercase_to_variants.items():
            if len(variants) > 1:
                print(f"\n  🔍 Trovate {len(variants)} varianti che differiscono solo per capitalizzazione per '{lowercase}':")
                for i, variant in enumerate(variants):
                    print(f"    {i+1}. {variant}")
                
                # Procedo automaticamente
                if True:
                    canonical_name = self._select_canonical_name(variants)
                    with self.db.driver.session() as session:
                        self._merge_entities(session, category_label, canonical_name, variants)
                    print(f"  ✅ Entità normalizzate sotto '{canonical_name}'")
                    
                    exact_matches_processed.update(variants)
        
        # Seconda fase: gestione delle varianti simili semanticamente
        remaining_entities = [e for e in entities if e not in exact_matches_processed]
        
        embeddings = {entity: self.dense_model.encode(entity.lower()) for entity in remaining_entities}
        
        merged_entities = {}
        processed = set()
        
        for entity in tqdm(remaining_entities, desc=f"  ↪ Analisi {category_label}", unit="entità"):
            if entity in processed:
                continue
                
            similar_entities = []
            entity_emb = embeddings[entity]
            
            for other_entity in remaining_entities:
                if other_entity != entity and other_entity not in processed:
                    other_emb = embeddings[other_entity]
                    similarity = np.dot(entity_emb, other_emb) / (np.linalg.norm(entity_emb) * np.linalg.norm(other_emb))
                    
                    if similarity > 0.90:
                        similar_entities.append(other_entity)
            
            if similar_entities:
                canonical_name = self._select_canonical_name([entity] + similar_entities)
                merged_entities[canonical_name] = [entity] + similar_entities
                processed.add(entity)
                processed.update(similar_entities)
            else:
                processed.add(entity)
        
        for canonical_name, variants in merged_entities.items():
            if len(variants) > 1:
                with self.db.driver.session() as session:
                    print(f"\n  🔍 Trovate {len(variants)} possibili varianti semanticamente simili per '{canonical_name}':")
                    for i, variant in enumerate(variants):
                        print(f"    {i+1}. {variant}")
                    
                    proceed = input(f"  ⚠️ Procedere con la normalizzazione? (y/n): ")
                    if proceed.lower() == 'y':
                        self._merge_entities(session, category_label, canonical_name, variants)
                        print(f"  ✅ Entità normalizzate sotto '{canonical_name}'")
                    else:
                        print(f"  ❌ Normalizzazione annullata per '{canonical_name}'")

    def _select_canonical_name(self, entities):
        """
        Seleziona il nome canonico tra un gruppo di entità simili,
        permettendo all'utente di scegliere tra le opzioni visualizzate.

        :param entities: Lista di entità simili
        :return: Il nome canonico scelto
        """
        if not entities:
            return None

        with self.db.driver.session() as session:
            entity_counts = {}
            for entity in entities:
                query = """
                MATCH (n)-[r]-(m)
                WHERE n.nome = $entity
                RETURN count(r) as count
                """
                result = session.run(query, entity=entity.lower())
                entity_counts[entity] = result.single()["count"]

        candidates = sorted(entities, key=lambda e: (-entity_counts[e], len(e)))
        print("Scegli il nome canonico tra le seguenti opzioni:")
        for i, candidate in enumerate(candidates):
            print(f"  {i+1}. {candidate} (count: {entity_counts[candidate]})")

        scelta = input("Inserisci il numero corrispondente alla tua scelta: ")
        try:
            idx = int(scelta) - 1
            if idx < 0 or idx >= len(candidates):
                print("Scelta non valida, verrà utilizzata l'opzione predefinita.")
                return candidates[0].lower()
            return candidates[idx].lower()
        except ValueError:
            print("Input non valido, verrà utilizzata l'opzione predefinita.")
            return candidates[0].lower()


    def _merge_entities(self, session, category_label, canonical_name, variants):
        """
        Unifica un gruppo di entità simili sotto un unico nome canonico.
        
        :param session: Sessione Neo4j
        :param category_label: Etichetta della categoria (Ingrediente, Tecnica, Licenza)
        :param canonical_name: Nome canonico da utilizzare
        :param variants: Lista di varianti da unificare
        """
        canonical_uuid = str(uuid.uuid4())
        session.run(
            f"""
            MERGE (c:{category_label} {{nome: $name}})
            ON CREATE SET c.uuid = $uuid
            """, 
            name=canonical_name.lower(), uuid=canonical_uuid
        )
        
        for variant in variants:
            if variant.lower() == canonical_name.lower():
                continue
                
            incoming_query = f"""
            MATCH (source)-[r]->(variant:{category_label} {{nome: $variant}})
            RETURN type(r) as type, source.uuid as sourceUuid
            """
            incoming = list(session.run(incoming_query, variant=variant.lower()))
            
            outgoing_query = f"""
            MATCH (variant:{category_label} {{nome: $variant}})-[r]->(target)
            RETURN type(r) as type, target.uuid as targetUuid
            """
            outgoing = list(session.run(outgoing_query, variant=variant.lower()))
            
            for rel in incoming:
                session.run(f"""
                MATCH (source) WHERE source.uuid = $sourceUuid
                MATCH (canonical:{category_label} {{nome: $canonical}})
                MERGE (source)-[:{rel['type']}]->(canonical)
                """, sourceUuid=rel["sourceUuid"], canonical=canonical_name.lower())
            
            for rel in outgoing:
                session.run(f"""
                MATCH (canonical:{category_label} {{nome: $canonical}})
                MATCH (target) WHERE target.uuid = $targetUuid
                MERGE (canonical)-[:{rel['type']}]->(target)
                """, canonical=canonical_name.lower(), targetUuid=rel["targetUuid"])
            
            session.run(f"""
            MATCH (variant:{category_label} {{nome: $variant}})
            DETACH DELETE variant
            """, variant=variant.lower())


    # -------------------------
    # Nuove funzioni per il vector indexing
    # -------------------------
    def insert_embedding(self, entity_name: str, entity_text: str, entity_id: str = None):
        embedding = self.dense_model.encode(entity_text)
        # Se entity_id è None, assegno un valore di default (ad esempio una stringa vuota)
        if entity_id is None:
            entity_id = ""
        with self.db.driver.session() as session:
            session.execute_write(self._create_entity_node_with_embedding, entity_name.lower(), entity_text, embedding, entity_id)
            
    @staticmethod
    def _create_entity_node_with_embedding(tx, entity_name, entity_text, embedding, entity_id):
        entity_uuid = str(uuid.uuid4())
        embedding_str = json.dumps(embedding.tolist())
        query = (
            "MERGE (e:Entity {nome: $entity_name, text: $entity_text, dish_mapping_id: $entity_id}) "
            "ON CREATE SET e.uuid = $entity_uuid, e.embedding = $embedding, e.nome = $entity_name"
        )
        tx.run(query, entity_name=entity_name, entity_text=entity_text, embedding=embedding_str, entity_uuid=entity_uuid, entity_id=entity_id)

if __name__ == "__main__":
    
    dataset_path = Path(config.DATASET_PATH + "/Menu")
    #indexer = RAGIndexer(clear_database=False)
    #indexer.normalize_similar_entities()
    #exit()
    
    dataset_path = Path(config.DATASET_PATH + "/Menu")
    indexer = RAGIndexer(clear_database=True)
    indexer.process_planets()
    indexer.process_codice_galattico()
    indexer.process_menu(dataset_path)
    indexer.process_manuale_cucina()
    #indexer.normalize_similar_entities()
    indexer.close()
