"""
Uses Huggingface Transformers Pipeline to generate using teapotai's teapotllm.
"""
from transformers import pipeline

DEFAULT_MODEL = ["teapotai/teapotllm", "34395ba99950379a1cc64cbb145176fd0339249a"]

class LLM:
    def __init__(self):
        self.pipe = pipeline("text2text-generation", model=DEFAULT_MODEL[0], revision=DEFAULT_MODEL[1])
    
    def generate(self, prompt: str) -> str:
        return self.pipe(prompt)[0]['generated_text']
    

if __name__ == "__main__":
    llm = LLM()
    prompt = "What is the capital of France?"
    response = llm.generate(prompt)
    print(f"Response: {response}")
