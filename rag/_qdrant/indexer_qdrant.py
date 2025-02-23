import fitz  # PyMuPDF
import json
import ollama
import numpy as np
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Union, Any
from tqdm import tqdm
import re
import emoji
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVector,
    SparseVectorParams,
    Modifier,
    PointStruct,
    TextIndexParams,
    IntegerIndexParams,
    TokenizerType
)
from fastembed import SparseTextEmbedding


class RAGIndexer:
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "menu_collection"
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.setup_qdrant()
        
        with open("dataset/Misc/dish_mapping.json", "r", encoding="utf-8") as f:
            self.dish_mapping = json.load(f)
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "menu_collection"
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.setup_qdrant()
        
        with open("dataset/Misc/dish_mapping.json", "r", encoding="utf-8") as f:
            self.dish_mapping = json.load(f)
    def setup_qdrant(self):
        """Configura la collezione su Qdrant."""
        if self.client.collection_exists(collection_name=self.collection_name):
            print(f"🗑️ Eliminazione della collezione esistente '{self.collection_name}'...")
            self.client.delete_collection(collection_name=self.collection_name)

        print(f"🆕 Creazione della collezione '{self.collection_name}'...")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(size=384, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "bm42": SparseVectorParams(modifier=Modifier.IDF)
            }
        )


        # Creazione degli indici full-text per i metadati con parametri ottimizzati
        field_params = {
            "nome_piatto": (2, 20),
            "ingredienti": (2, 20),
            "tecniche": (2, 20),
            "chef": (2, 25),
            "ristorante": (2, 25),
            "testo_piatto": (3, 30),
            "testo_generale": (3, 30),
        }
        
        self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="id_piatto",
                field_schema=IntegerIndexParams(
                    type="integer",
                ),
            )
        
        for field, (min_len, max_len) in field_params.items():
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD,
                    min_token_len=min_len,
                    max_token_len=max_len,
                    lowercase=True,
                ),
            )

    def extract_plain_text_from_pdf(self, pdf_path: Path) -> List[str]:
        """
        Estrae il testo da un PDF e aggiunge un marker speciale (|||TITOLO|||) prima di ogni piatto per facilitarne la suddivisione.
        
        :param pdf_path: Percorso del file PDF
        :param dishes: Lista dei piatti da identificare nel testo
        :return: Testo con marker inseriti
        """
        text = ""

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    # Controlla se nel testo della pagina è presente un titolo di piatto
                    for dish in dishes:
                        # Usa regex per trovare il piatto come titolo (inizio riga o seguito da punto/spazio)
                        pattern = rf"(^|\n|\s)({re.escape(dish)})(\s|\.|$)"
                        page_text = re.sub(pattern, r"\1|||TITOLO||| \2", page_text, flags=re.IGNORECASE)

                    text += page_text + "\n"

        return text

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

    def identify_titles(self, extracted_data: List[Dict[str, str]]) -> List[str]:
        """
        Identifica i titoli basandosi sulla dimensione del font.

        :param extracted_data: Lista di dizionari contenenti testo e metadati.
        :return: Lista dei titoli identificati.
        """
        if not extracted_data:
            return []

        # Trova la dimensione media dei caratteri (per distinguere i titoli)
        font_sizes = [item["size"] for item in extracted_data]
        avg_font_size = sum(font_sizes) / len(font_sizes)

        # Consideriamo un titolo se ha un font più grande della media
        titles = [item["text"] for item in extracted_data if item["size"] > avg_font_size * 1.3]

        return titles


    def insert_markers_and_split(self, extracted_data: List[Dict[str, str]], titles: List[str]) -> Dict[str, List[str]]:
        """
        Suddivide il testo tra un titolo e l'altro in base ai titoli riconosciuti nel documento.

        :param extracted_data: Lista di dizionari con testo e metadati.
        :param titles: Lista dei titoli identificati.
        :return: Dizionario con i titoli come chiavi e le sezioni di testo associate.
        """
        sections = {}
        current_title = None

        for i, item in enumerate(extracted_data):
            line_text = item["text"].strip()

            # Controllo che il titolo sia effettivamente isolato e non parte di una frase
            if line_text in titles and (i == 0 or extracted_data[i - 1]["text"].strip() == ""):
                if line_text in sections:  # Se il titolo è già stato trovato, significa che c'è un errore di parsing
                    continue  
                current_title = line_text
                sections[current_title] = []
            elif current_title:
                sections[current_title].append(line_text)

        return sections


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
                if line_text.strip() in self.dish_mapping and current_title != line_text:
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

    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, str]]:
        """Estrae testo e identifica elementi formattati (grassetto/corsivo)."""
        extracted_text = []
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    for block in page.get_text("dict")["blocks"]:
                        for line in block.get("lines", []):
                            for word in line["spans"]:
                                text = word["text"].strip()
                                if text:
                                    style = "normal"
                                    # Controlla i flag per grassetto e corsivo
                                    if word.get("flags", 0) & 2:  # Grassetto
                                        style = "bold"
                                    elif word.get("flags", 0) & 1:  # Corsivo
                                        style = "italic"
                                    extracted_text.append({"text": text, "style": style})
        except Exception as e:
            print(f"⚠️ Errore nell'estrazione PDF {pdf_path}: {e}")
        return extracted_text

    def find_dishes_in_text(self, text_data: List[Dict[str, str]]) -> List[str]:
        """Trova i nomi dei piatti nel testo del PDF."""
        text = " ".join([t["text"] for t in text_data])
        with open("dataset/Misc/dish_mapping.json", "r", encoding="utf-8") as f:
            dish_mapping = json.load(f)
        return [dish for dish in dish_mapping.keys() if dish in text]
    
    def find_dishes_in_plain_text(self, text_data: List[Dict[str, str]]) -> List[str]:
        """Trova i nomi dei piatti nel testo del PDF."""
        with open("dataset/Misc/dish_mapping.json", "r", encoding="utf-8") as f:
            dish_mapping = json.load(f)
        return [dish for dish in dish_mapping.keys() if dish in text_data]

    def find_dishes_in_sections(self, sections: Dict[str, List[str]], dish_mapping_path: str = "dataset/Misc/dish_mapping.json") -> List[str]:
        """
        Trova i nomi dei piatti nel testo suddiviso in sezioni.

        :param sections: Dizionario con titoli di sezione come chiavi e testo come valori.
        :param dish_mapping_path: Percorso del file JSON contenente la mappatura dei piatti.
        :return: Lista dei nomi dei piatti trovati nel testo.
        """
        with open(dish_mapping_path, "r", encoding="utf-8") as f:
            dish_mapping = json.load(f)

        found_dishes = []
        for dish in dish_mapping.keys():
            for section_text in sections.values():
                if any(dish in text for text in section_text):  # Controlla se il piatto è menzionato nella sezione
                    found_dishes.append(dish)
                    break  # Se il piatto è stato trovato in una sezione, non serve cercarlo nelle altre
        
        return found_dishes

    def split_text_by_dishes(self, sections: Dict[str, List[str]], dishes: List[str]) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Divide il testo in sezioni specifiche per ogni piatto e separa il testo generale che non appartiene a nessun piatto.

        :param sections: Dizionario con titoli come chiavi e testo come valori (lista di stringhe).
        :param dishes: Lista dei nomi dei piatti da cercare.
        :return: Un dizionario con i piatti come chiavi e le sezioni corrispondenti, e una lista con il testo generale.
        """
        dish_sections = {dish: [] for dish in dishes}
        general_text = []

        for section_title, section_text in sections.items():
            matched = False
            for dish in dishes:
                if any(dish in line for line in section_text):
                    dish_sections[dish].extend(section_text)
                    matched = True
                    break  # Se il titolo appartiene a un piatto, non serve cercarlo negli altri
            
            if not matched:
                general_text.extend(section_text)  # Se la sezione non è un piatto, va nel testo generale

        return dish_sections, general_text


    def extract_chef_from_intro(self, text: str) -> str:
        prompt = (
            "Estrarre il nome del ristorangte e dello chef dal seguente testo: \n"
            f"{text}\n\n"
            "Restituisci solo il nome del ristorante e il nome completo dello chef, senza testo aggiuntivo."
            "Fornisci il risultato in formato JSON con le chiavi 'ristorante' e 'nome_chef'.\n"

        )
        
        response = ollama.chat(
            model="qwen2.5",
            format="json",
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            content = json.loads(response["message"]["content"])
            restaurant_name = content.get("ristorante", "Non specificato").strip()
            chef_name = content.get("nome_chef", "Non specificato").strip()
            return restaurant_name,chef_name
        except json.JSONDecodeError:
            return "Non specificato"


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

        response = ollama.chat(
            model="qwen2.5",
            format="json",
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            content = json.loads(response["message"]["content"])
            return {
                "ingredienti": content.get("ingredienti", "Non specificato"),
                "tecniche": content.get("tecniche", "Non specificato")
            }
        except json.JSONDecodeError:
            return {"ingredienti": "Non specificato", "tecniche": "Non specificato"}


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
                        t["text"].strip() for t in region_tokens
                        if isinstance(t, dict) and "style" in t and t["style"] in ["bold", "italic"] and t["text"].strip()
                    ]
                else:
                    combined_text = " ".join(
                        t["text"] for t in region_tokens if isinstance(t, dict) and "text" in t
                    )
                    if "\n" in combined_text:
                        extracted = [line.strip() for line in combined_text.split("\n") if line.strip()]
                    elif len(region_tokens) > 1:
                        extracted = [t["text"].strip() for t in region_tokens if isinstance(t, dict) and "text" in t and t["text"].strip()]
                    elif region_tokens:
                        token_text = region_tokens[0]["text"].strip() if isinstance(region_tokens[0], dict) and "text" in region_tokens[0] else ""
                        if "\n" in token_text:
                            extracted = [line.strip() for line in token_text.split("\n") if line.strip()]
                        else:
                            extracted = re.findall(r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){1,3})\b', token_text)
                    else:
                        extracted = []
                return extracted
            else:
                # Fallback: applica una regex all'intero testo della sezione
                combined_text = " ".join(
                    token["text"] for token in tokens if isinstance(token, dict) and "text" in token
                )
                candidates = re.findall(r'\b(?:[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})\b', combined_text)
                return [c.strip() for c in candidates if c.strip()]

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
                llm_items = [str(item).strip() for item in llm_value if str(item).strip()]
            elif isinstance(llm_value, str) and llm_value.lower() != "non specificato":
                llm_items = [item.strip() for item in llm_value.split(",") if item.strip()]
            merged = list(set(rule_list + llm_items))
            return ", ".join(merged) if merged else "Non specificato"


        merged_ingredienti = merge_category(rules_data.get("ingredienti", []), llm_data.get("ingredienti", "Non specificato"))
        merged_tecniche = merge_category(rules_data.get("tecniche", []), llm_data.get("tecniche", "Non specificato"))
        merged_chef = global_chef if global_chef and global_chef.strip() else "Non specificato"
        return {"ingredienti": merged_ingredienti, "tecniche": merged_tecniche, "chef": merged_chef}

    def verify_extraction(self, extracted_data: Dict[str, str], section_text: str) -> Dict[str, str]:
        """
        Verifica se gli ingredienti e le tecniche estratte sono effettivamente presenti nel testo della sezione.
        Se un elemento non viene trovato, viene rimosso e segnalato.
        """
        def verify_items(category: str, items: str) -> str:
            return items
        
        
        
        
            if items == "Non specificato":
                return items
            valid_items = []
            removed_items = []
            for item in items.split(", "):
                if item.lower() in section_text.lower():
                    valid_items.append(item)
                else:
                    removed_items.append(item)
            if removed_items:
                print(f"⚠️ Elementi rimossi dalla categoria '{category}': {', '.join(removed_items)}")
                exit()
            return ", ".join(valid_items) if valid_items else "Non specificato"
        extracted_data["ingredienti"] = verify_items("ingredienti", extracted_data["ingredienti"])
        extracted_data["tecniche"] = verify_items("tecniche", extracted_data["tecniche"])
        
        return extracted_data

    def process_pdf(self, pdf_path: Path):
        """Estrai i piatti da un PDF e indicizza ingredienti, tecniche, chef e testo su Qdrant."""
        print(f"\n📄 Analizzando {pdf_path.name}...")
        #text_data = self.extract_text_from_pdf(pdf_path)
        #text_data = self.extract_plain_text_from_pdf(pdf_path)
        
        extracted_data = self.extract_text_with_metadata(pdf_path)

        #titles = self.identify_titles(extracted_data)

        #sections = self.insert_markers_and_split(extracted_data, titles)
        
        sections = self.extract_sections_from_text(extracted_data)
        
        #print(sections)
        #exit()
        
        #found_dishes = self.find_dishes_in_text(text_data)
        #found_dishes = self.find_dishes_in_plain_text(text_data)
        #found_dishes = self.find_dishes_in_sections(sections)
        
        #sections, general_text = self.split_text_by_dishes(sections, found_dishes)

        general_text = " ".join(sections['general_text'])
        
        # Estrae il nome dello chefqq\1 dal testo introduttivo (general_text)
        global_ristorante, global_chef = self.extract_chef_from_intro(general_text)
        # Estrae il nome del ristorante dal nome del file (stem)
        #ristorante = pdf_path.stem
        category = pdf_path.parent.stem
    
        for dish, section in sections.items():
            if not section:
                continue
        
            section_tokens = section
            if section and section[0].strip() == dish:
                section_tokens = section[1:]
            
            # Estrazione via regole per ingredienti e tecniche
            rules_data = self.extract_by_rules(section_tokens)
            
            # Estrazione tramite LLM per ingredienti e tecniche
            llm_data = self.extract_with_llm(section_tokens)

            """           
            if(dish=="Cosmic Rhapsody"):
                print(section_tokens)
                print("================")
                print(llm_data)
                print("================")
                exit()
            else:
                continue
            """

            # Merge dei risultati
            extracted_data = self.merge_extraction_results(rules_data, llm_data, global_chef)
            section_text = " ".join(section) 
            #extracted_data = self.verify_extraction(extracted_data, section_text)
            
            dish_text = " ".join(section)
            if(dish_text.strip() == ""):
                continue
            
            print(f"🥗 {dish} - Ingredienti: {extracted_data['ingredienti']}")
            print(f"🔥 Tecniche: {extracted_data['tecniche']}")
            print(f"👨‍🍳 Chef: {extracted_data['chef']}")
            
            # Creazione degli embedding
            dense_embedding = self.dense_model.encode(dish_text).tolist()
            #sparse_text = f"Ingredienti: {extracted_data['ingredienti']}, Tecniche: {extracted_data['tecniche']}"
            
            sparse_text = (
                f"{dish}. Questa è una ricetta con ingredienti come {extracted_data['ingredienti']}. "
                f"Le tecniche di preparazione utilizzate sono {extracted_data['tecniche']}."
            )


            sparse_embedding = list(self.sparse_model.embed([sparse_text]))[0]
            
            dish_id = uuid.uuid5(uuid.NAMESPACE_DNS, dish).hex
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=dish_id,
                        vector={
                            "dense": dense_embedding,
                            "bm42": SparseVector(
                                values=sparse_embedding.values.tolist(),
                                indices=sparse_embedding.indices.tolist()
                            )
                        },
                        payload={
                            "category": category,
                            "id_piatto": self.dish_mapping.get(dish, ""),
                            "nome_piatto": dish,
                            "ingredienti": extracted_data["ingredienti"],
                            "tecniche": extracted_data["tecniche"],
                            "chef": extracted_data["chef"],
                            "ristorante": global_ristorante,
                            "testo_piatto": dish_text,
                            "source": str(pdf_path)
                        }
                    )
                ]
            )

        # Indicizza il testo generale del PDF (se presente)
        if general_text:
            general_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(pdf_path)).hex
            dense_embedding = self.dense_model.encode(general_text).tolist()
            sparse_embedding = list(self.sparse_model.embed([general_text]))[0]

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=general_id,
                        vector={
                            "dense": dense_embedding,
                            "bm42": SparseVector(
                                values=sparse_embedding.values.tolist(),
                                indices=sparse_embedding.indices.tolist()
                            )
                        },
                        payload={
                            "testo_generale": general_text,
                            "ristorante": global_ristorante,
                            "source": str(pdf_path)
                        }
                    )
                ]
            )

    def process_directory(self, dataset_path: Path):
        """Processa tutti i PDF nella cartella."""
        pdf_files = list(dataset_path.rglob("*.pdf"))
        if not pdf_files:
            print("⚠️ Nessun PDF trovato nella cartella.")
            return
        for pdf_file in tqdm(pdf_files, desc="📄 Analisi PDF"):
            self.process_pdf(pdf_file)

    def generate_report(self):
        """Genera un report sull'indicizzazione."""
        # Carica la mappatura dei piatti
        with open("dataset/Misc/dish_mapping.json", "r", encoding="utf-8") as f:
            dish_mapping = json.load(f)

        total_dishes = len(dish_mapping)
        found_dishes = 0
        report = {}

        # Recupera i dati indicizzati da Qdrant
        results = self.client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=True
        )

        # Costruisce il report basandosi sui dati presenti in Qdrant
        for hit in results[0]:
            nome_piatto = hit.payload.get("nome_piatto", "Sconosciuto")
            ingredienti = hit.payload.get("ingredienti", "").split(",")
            tecniche = hit.payload.get("tecniche", "").split(",")
            chef = hit.payload.get("chef", "Non specificato")
            if nome_piatto in dish_mapping:
                found_dishes += 1
            report[nome_piatto] = {
                "ingredienti": len([i for i in ingredienti if i.strip()]),
                "tecniche": len([t for t in tecniche if t.strip()]),
                "chef": chef
            }

        print("\n📊 Report sull'Indicizzazione")
        print(f"🍽️ Piatti trovati: {found_dishes}/{total_dishes}")

        # Mostra i piatti che hanno meno di 2 ingredienti
        print("\n🔍 Piatti con meno di 2 ingredienti:")
        found_with_few_ingredients = False
        for dish, details in report.items():
            if details["ingredienti"] < 2:
                print(f" - {dish}: {details['ingredienti']} ingredienti | Chef: {details['chef']}")
                found_with_few_ingredients = True
        if not found_with_few_ingredients:
            print("Tutti i piatti hanno almeno 2 ingredienti.")

        # Identifica i piatti presenti nella mappatura ma non trovati in Qdrant
        not_found_dishes = [dish for dish in dish_mapping.keys() if dish not in report]
        print("\n🚫 Piatti non trovati:")
        if not_found_dishes:
            for dish in not_found_dishes:
                print(f" - {dish}")
        else:
            print("Tutti i piatti sono stati trovati.")

        # Elenca i piatti e il nome dello chef associato
        #print("\n👨‍🍳 Elenco dei piatti con il nome dello chef:")
        #for dish, details in report.items():
        #    print(f" - {dish}: Chef -> {details['chef']}")


def main():
    dataset_path = Path("dataset/Menu")
    indexer = RAGIndexer()
    indexer.process_directory(dataset_path)
    indexer.generate_report()


if __name__ == "__main__":
    main()
