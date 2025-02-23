import re
import json
from utils.logger import Logger

class ResponseParser:
    def __init__(self, verbose=True):
        self.logger = Logger(verbose)

    def extract_next_action(self, response: str) -> list:
        """Estrae la prossima azione dalla risposta LLM."""
        steps = []
        pattern = re.compile(r"Thought:\s*(.*?)\s*Action:\s*```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

        for match in pattern.finditer(response):
            thought = match.group(1).strip()
            action_blob = match.group(2).strip()
            try:
                json_data = json.loads(action_blob)
                steps.append({"Thought": thought, "Action": json_data, "Observation": None})
                return steps
            except json.JSONDecodeError:
                self.logger.error("❌ JSON non valido nel blocco Action.")

        return steps

    def parse_final_answer(self, response: str):
        """Estrae la risposta finale dall'output LLM."""
        final_answer_match = re.search(r'Final Answer:\s*([\s\S]*?)(?=\n\n|$)', response)
        return final_answer_match.group(1).strip() if final_answer_match else None
