import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:14b")
DATASET_PATH = os.getenv("DATASET_PATH", "dataset")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "output")
FEEDBACK_FILE = os.getenv("FEEDBACK_FILE", "feedback.json")

AGENT_MAX_ITERATIONS = 20