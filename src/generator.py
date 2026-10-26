from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict


class AnswerGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B"):
        print(f"Kan-loadiw l'Model '{model_name}'")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto"
            )
        print("L'Model load...!")

    def build_prompt(self, question: str, retrieved_chunks: List[Dict]) -> str:
        context_text = ""
        for i, chunk in enumerate(retrieved_chunks):
            context_text += f"\n--- Source {i+1} ---\n{chunk['text']}\n"

        #  promte
        prompt = f"""You are a helpful and accurate assistant. Answer the user's question based ONLY on the provided context below. If the answer is not in the context, say "I don't know".
        Context: {context_text}
        Question: {question}
        Answer:"""
        return prompt
    def generate(self, question: str, retrieved_chunks: List[Dict]) -> str:
        prompt = self.build_prompt(question, retrieved_chunks)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            # temperature=0.1,
            do_sample=False
        )

        input_length = inputs['input_ids'].shape[1]
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        return response.strip()
# ==========================================
if __name__ == "__main__":
    generator = AnswerGenerator()
    
    fake_question = "How to configure OpenAI compatible server?"
    fake_chunks = [
        {"text": "To configure the OpenAI compatible server in vLLM, you need to use the command: python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m."}
    ]
    
    print("\n L'Model kayfker...")  
    answer = generator.generate(fake_question, fake_chunks)
    print(f"\nwen Jawb:\n{answer}")
