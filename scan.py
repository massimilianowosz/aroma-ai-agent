import giskard
import pandas as pd
from utils.llm import LLM,Ollama
from utils.logger import Logger
from utils import config
from agent.agent import Agent

class GiskardWrapper:
    def __init__(self, agent, questions_file: str):
        self.agent = agent
        self.questions_file = questions_file
    
    def load_questions(self):
        """
        Carica il file CSV con le domande.
        """
        try:
            df = pd.read_csv(self.questions_file)
            # Assumiamo che la colonna contenente le domande si chiami 'domanda'
            if "domanda" not in df.columns:
                raise ValueError("Il CSV non contiene una colonna 'domanda'.")
            return df
        except Exception as e:
            raise ValueError(f"Errore durante il caricamento del file: {str(e)}")

    def model_predict(self, df: pd.DataFrame):
        """
        Wrapper per fare la previsione del modello, utilizzando le domande nel dataset.
        """
        responses = []
        for prompt in df["domanda"]:  # Assicurati che la colonna si chiami 'domanda'
            final_answer = self.agent.run(prompt)
            responses.append(final_answer)
        return responses
    
    def scan(self):
        """
        Esegui la scansione del modello tramite Giskard.
        """
        # Carica il dataset con le domande
        df = self.load_questions()

        # Crea il modello di Giskard utilizzando il wrapper
        giskard_model = giskard.Model(
            model=self.model_predict,
            model_type="text_generation",  # O il tipo che il tuo modello supporta
            name="Agente con Giskard",
            description="Scansione automatica per l'agente",
            feature_names=["domanda"],  # Cambia qui anche nella lista delle feature
        )

        # Esegui la scansione
        scan_results = giskard.scan(giskard_model)
        
        # Mostra o salva i risultati della scansione
        return scan_results

# Esegui la scansione
if __name__ == "__main__":
    giskard.llm.set_llm_model("ollama/qwen2.5:14b", disable_structured_output=True, api_base="http://localhost:11434")
    giskard.llm.set_embedding_model("ollama/nomic-embed-text", api_base="http://localhost:11434")

    SYSTEM_PROMPT = """
    Rispondi alla seguente domanda nel modo più accurato possibile utilizzando i dati a tua disposizione e gli strumenti disponibili.

    Formato da seguire:

    1. Question: [domanda da rispondere]

    2. Per ogni passaggio necessario:
    - Thought: spiega brevemente il prossimo passo da fare
    - Action: 
        ```json
        {{
        "tool": "nome_strumento",
        "params": {{
            "arg1": [],
            "arg2": ...
        }}
        }}
        ```                
    - Observation (optional): esito dell'azione. Non inventare observation ma utilizza sempre i dati reali.

    3. Conclusione:
    Dopo ciascuna Observation analizza i dati a tua disposizione per rispondere alla domanda dell'utente.
    Thought: Ho trovato la risposta
    Final Answer: [risposta completa che include tutti gli elementi rilevanti]

    Note:
    - Usa una Action alla volta
    - Se generi una Action non generare Observation e Final Answer inventate.
    """

    agent = Agent(system_prompt=SYSTEM_PROMPT, verbose=True, interactive=True, guard=True)

    # Percorso al file CSV contenente le domande
    questions_file = "dataset/domande.csv"  # Modifica il percorso del file qui

    # Avvia il wrapper Giskard
    giskard_wrapper = GiskardWrapper(agent, questions_file)

    # Esegui la scansione
    scan_results = giskard_wrapper.scan()

    # Visualizza o salva i risultati
    print(scan_results)
    scan_results.to_html("scan_results.html")