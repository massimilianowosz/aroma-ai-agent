import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import ollama
from openai import OpenAI as OpenAIClient
from groq import Groq as GroqClient

load_dotenv()

class LLM(ABC):
    """
    Interfaccia astratta per un Large Language Model (LLM).
    Ogni implementazione dovrà definire il metodo generate.
    """
    @abstractmethod
    def generate(self, messages: list, format: str = "") -> str:
        """
        Genera una risposta basata su una lista di messaggi.
        
        :param messages: Lista di messaggi da inviare al modello.
        :param format: (Opzionale) Formato della risposta.
        :return: Risposta generata come stringa.
        """
        pass

class Ollama(LLM):
    def __init__(self, model: str):
        self.model = model

    def generate(self, messages: list, format: str = "") -> str:
        # Chiama il modello di Ollama e restituisce la risposta
        response = ollama.chat(model=self.model, messages=messages, format=format)
        return response.get("response", response.get("message", {}).get("content", ""))

class Groq(LLM):
    def __init__(self, model: str):
        self.model = model
        # Inizializza il client Groq utilizzando la variabile d'ambiente GROQ_API_KEY
        self.client = GroqClient(api_key=os.environ.get("GROQ_API_KEY"))

    def generate(self, messages: list, format: str = "") -> str:
        # Richiama l'API di Groq per ottenere il completamento della chat
        completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model
        )
        return completion.choices[0].message.content

class OpenAI(LLM):
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAIClient(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )


    def generate(self, messages: list, format: str = "") -> str:
        # Genera una risposta utilizzando l'API di OpenAI
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content
