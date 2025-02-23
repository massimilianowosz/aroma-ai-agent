from agent.tools.cypher_feedback_tool import cypher_feedback_tool
from utils.logger import Logger

class ToolExecutor:
    def __init__(self, tools={}, interactive=False, verbose=True):
        self.logger = Logger(verbose)
        self.tools = tools
        if interactive:
            self.tools["cypher_feedback_tool"] = cypher_feedback_tool

    def execute(self, tool_name, tool_params):
        """Esegue un tool specifico e restituisce l'output."""
        tool = self.tools.get(tool_name)
        if tool:
            result = tool(**tool_params)
            return result
        else:
            self.logger.error(f"❌ Tool '{tool_name}' non trovato.")
            return None
