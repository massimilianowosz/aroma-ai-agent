from neo4j import GraphDatabase
import logging
from utils import config

logger = logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    
    def close(self):
        self.driver.close()
    
    def run_query(self, query, **params):
        with self.driver.session() as session:
            return session.run(query, **params)
    
    def execute_write(self, func, *args, **kwargs):
        with self.driver.session() as session:
            return session.execute_write(func, *args, **kwargs)