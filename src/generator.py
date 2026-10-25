from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict


class AnswerGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        print(f"⏳ Kan-loadiw l'Model '{model_name}'")
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="cpu"
        )
    print("L'Model load...!")

    def build_prompt(self, question: str, retrieved_chunks: List[Dict]) -> str:
        context_text = ""
        for i, chunk in enumerate(retrieved_chunks):
            context_text += f"\n--- Source {i+1} ---\n{chunk['text']}\n"

        // promte
        prompt = f"""You are a helpful and accurate assistant. Answer the user's question based ONLY on the provided context below. If the answer is not in the context, say "I don't know".

        Context:
        {context_text}

        Question: {question}
        Answer:"""
            return prompte
    def generate(self, question: str, retrieved_chunks: List[Dict]) -> str:
         prompt = self.build_prompt(question, retrieved_chunks)
        inputs = self.tokenizer(prompt, return_tensors="pt")
