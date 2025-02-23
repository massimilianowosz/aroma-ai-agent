from agent.agent import Agent
from utils import config
from utils.llm import Ollama

#llm_instance = OpenAI(model="gpt-4o-mini")

agent = Agent(verbose=True, interactive=True, guard=False)
query = "Quali sono i piatti che includono le Chocobo Wings come ingrediente?"
#query = "Quali piatti preparati con la tecnica Grigliatura a Energia Stellare DiV?"
#query = "Quali piatti includono Lattuga Namecciana e Carne di Mucca ma non contengono Teste di Idra?"
#query = "Quali piatti leggendari dell'universo utilizzano Foglie di Nebulosa, Amido di Stellarion e Uova di Fenice nella loro preparazione?"
#query = "Quali piatti, preparati in un ristorante su Asgard, richiedono la licenza LTK non base e utilizzano Carne di Xenodonte?"
#query = "Quali sono i piatti della galassia che contengono Latte+?"
#query = "Quali sono i piatti che includono i Sashimi di Magikarp?"
#query = "Quali sono i piatti che combinano Carne di Balena spaziale, Teste di Idra, Spaghi del Sole e Carne di Xenodonte nella loro preparazione?"

structured_prompt = """
Rispondi con SOLO una lista di ID numerici separati da virgola. 
Ad esempio: 12,34,56
Non aggiungere testo aggiuntivo. Solo numeri e virgole.
"""

final_answer = agent.run(query, structured_prompt)
print(final_answer)