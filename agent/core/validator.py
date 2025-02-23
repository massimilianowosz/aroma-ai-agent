import json
from utils.logger import Logger
from agent.tools.validate_ids_tool import validate_ids_tool

class ResponseValidator:
    def __init__(self, llm, verbose=True):
        self.logger = Logger(verbose)
        self.llm = llm

    def validate_tool_results(self, tool_results, question: str) -> list:
        """Valida i risultati del tool confrontandoli con la domanda."""
        validated_data = []

        parsed_results = json.loads(tool_results) if isinstance(tool_results, str) else tool_results
        matches = parsed_results.get('matches', [])

        for record in matches:
            record_str = json.dumps(record) if not isinstance(record, str) else record
            validation_prompt = f"""
            La risposta soddisfa la domanda in modo completo ed accurato? 
            Rispondi con "True" se lo fa, o "False" altrimenti.

            Domanda:
            {question}

            Questo record risponde alla domanda?
            {record_str}
            """

            messages = [{"role": "user", "content": validation_prompt}]
            llm_response = self.llm.generate(messages=messages)

            if "True" in llm_response:
                validated_data.append(record)
            else:
                self.logger.error(f"❌ Record non valido: {record}")

        return validated_data
