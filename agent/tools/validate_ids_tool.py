import json
from agent.tools.base import tool

@tool
def validate_ids_tool(ids: str) -> str:
    """
    Valida una lista di ID numerici (separati da virgola) confrontandoli con il dish mapping presente in JSON.
    
    Parametri:
      - ids: stringa contenente ID separati da virgola, ad es. "78,256,190"
      
    Restituisce:
      - Una stringa contenente solo gli ID validi (quelli presenti nei valori del dish mapping), separati da virgola.
    """
    try:
        mapping_file = "dataset/Misc/dish_mapping.json"
        # Carica il file JSON del dish mapping
        with open(mapping_file, "r", encoding="utf-8") as f:
            dish_mapping = json.load(f)
        
        # Crea un insieme dei valori validi (convertiti in stringa)
        valid_ids_set = {str(value) for value in dish_mapping.values()}
        
        # Filtra gli ID forniti: mantieni solo quelli presenti nel dish mapping
        valid_ids = [id_str.strip() for id_str in ids.split(",") if id_str.strip() in valid_ids_set]
        
        return ",".join(valid_ids)
    except Exception as e:
        return f"Errore durante la validazione degli id: {e}"