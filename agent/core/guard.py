from transformers import pipeline

class PromptGuard:
    def __init__(self):
        self.classifier = pipeline("text-classification", model="meta-llama/Prompt-Guard-86M")
        self.max_length = 512

    def split_text(self, text: str):
        return [text[i:i + self.max_length] for i in range(0, len(text), self.max_length)]

    def classify(self, text: str):
        segments = self.split_text(text)
        
        for segment in segments:
            result = self.classifier(segment)
            if result[0]['label'] != 'BENIGN':
                return False
        
        return True