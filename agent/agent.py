import json
from utils import config
from utils.llm import LLM
from utils.logger import Logger
from agent.core.executor import ToolExecutor
from agent.core.parser import ResponseParser
from agent.core.validator import ResponseValidator
from agent.core.guard import PromptGuard

from agent.tools.cypher_search_tool import cypher_search_tool
from agent.tools.vector_search_tool import vector_search_tool
from agent.tools.validate_ids_tool import validate_ids_tool

from utils.llm import Ollama
from utils.prompts import SYSTEM_PROMPT

class Agent:
    def __init__(self, system_prompt: str = "", verbose: bool = True, interactive: bool = False, tools: dict = {}, guard: bool = False):
        self.messages = []
        self.verbose = verbose
        self.guard = guard
        self.logger = Logger(verbose)
        if not tools:
            tools = {
                "cypher_search_tool": cypher_search_tool,
                "vector_search_tool": vector_search_tool,
                "validate_ids_tool": validate_ids_tool,
            }
        self.tools = tools

        self.executor = ToolExecutor(self.tools, interactive, verbose)
        self.parser = ResponseParser(verbose)
        self.llm = Ollama(config.MODEL_NAME)

        self.validator = ResponseValidator(self.llm, verbose)
        self.system_prompt = system_prompt or self._default_system_prompt()
        
    def _default_system_prompt(self) -> str:
        tools_descriptions = "\n\n".join([tool.to_string() for tool in self.executor.tools.values()])
        return SYSTEM_PROMPT.format(tools_descriptions=tools_descriptions)
    
    def run(self, prompt: str, output_format: str = "") -> str:
        if self.guard:
            if( not PromptGuard().classify(prompt) or not PromptGuard().classify(output_format)):
                return "Non posso eseguire questa azione."
            
        # Concatena il formato richiesto al system prompt
        self.messages = [{"role": "system", "content": self.system_prompt+output_format}]
        self.messages.append({"role": "user", "content": prompt})

        iteration = 0
        while iteration < config.AGENT_MAX_ITERATIONS:
            response = self.llm.generate(messages=self.messages)
            #self.logger.debug(response)
            
            steps = self.parser.extract_next_action(response)
            if not steps:
                final_ans = self.parser.parse_final_answer(response)
                #Not CoT prompt
                if not final_ans:
                    return response
                return final_ans

            step = steps[0]
            self.logger.step(iteration + 1, "Processing step")
            self.logger.thought(step["Thought"])

            if step["Action"] and "tool" in step["Action"]:
                step["Observation"] = None
                tool_result = self.executor.execute(step["Action"]["tool"], step["Action"].get("params", {}))
                observation = f"Tool '{step['Action']['tool']}' returned:\n{tool_result}."

                if tool_result and "Nessun piatto trovato" in tool_result:
                    return '0'

                self.logger.action(step["Action"])
                self.logger.observation(observation)

                self.messages.append({"role": "assistant", "content": observation})

            iteration += 1

        self.logger.error("Numero massimo di iterazioni raggiunto")
        return "Non sono riuscito a trovare una risposta definitiva."
