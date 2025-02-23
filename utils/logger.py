from typing import Dict, Any
from datetime import datetime
from colorama import Fore, Style, init as colorama_init
import json

colorama_init()

class Logger:
    _verbose: bool = True  # Verbosità abilitata di default

    def __init__(self, verbose: bool = True):
        self._verbose = verbose

    def set_verbose(self, verbose: bool):
        self._verbose = verbose
        
    def step(self, step_num: int, message: str):
        if self._verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"{Fore.CYAN}[Step {step_num} | {timestamp}] {message}{Style.RESET_ALL}")

    def thought(self, thought: str):
        if self._verbose:
            print(f"{Fore.GREEN}🤔 Thought: {thought}{Style.RESET_ALL}")

    def action(self, action: Dict):
        if self._verbose:
            print(f"{Fore.YELLOW}⚡ Action: {json.dumps(action, indent=2)}{Style.RESET_ALL}")

    def observation(self, obs: Any):
        if self._verbose:
            print(f"{Fore.MAGENTA}👁️ Observation: {obs}{Style.RESET_ALL}")

    def answer(self, answer: str):
        if self._verbose:
            print(f"{Fore.GREEN}✅ Final Answer: {answer}{Style.RESET_ALL}")

    def error(self, error: str):
        if self._verbose:
            print(f"{Fore.RED}❌ Error: {error}{Style.RESET_ALL}")
    
    def debug(self, debug: str):
        if self._verbose:
            print(f"{Fore.LIGHTWHITE_EX}Debug: {debug}{Style.RESET_ALL}")
    
    def info(self, message: str):
        if self._verbose:
            print(f"{Fore.BLUE}ℹ️ Info: {message}{Style.RESET_ALL}")
