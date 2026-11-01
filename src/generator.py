from typing import Any, Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer


class AnswerGenerator:
    """Generate answers using a causal language model."""

    def __init__(
        self, model_name: str = "Qwen/Qwen2.5-0.5B"
    ) -> None:
        """Initialise the generator and load the model.

        Args:
            model_name: HuggingFace model identifier.
        """
        print(f"Kan-loadiw l'Model '{model_name}'")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )
        print("L'Model load...!")

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        """Build a prompt from the question and context.

        Args:
            question: The user question.
            retrieved_chunks: Retrieved context chunks.

        Returns:
            The formatted prompt string.
        """
        context_text = ""
        for i, chunk in enumerate(retrieved_chunks):
            context_text += (
                f"\n--- Source {i+1} ---\n"
                f"{chunk['text']}\n"
            )

        #  promte
        prompt = (
            "You are a helpful and accurate assistant."
            " Answer the user's question based ONLY on"
            " the provided context below. If the answer"
            " is not in the context, say"
            ' "I don\'t know".\n'
            f"        Context: {context_text}\n"
            f"        Question: {question}\n"
            "        Answer:"
        )
        return prompt

    def generate(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        """Generate an answer for the given question.

        Args:
            question: The user question.
            retrieved_chunks: Retrieved context chunks.

        Returns:
            The generated answer text.
        """
        prompt = self.build_prompt(
            question, retrieved_chunks
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt"
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
        )

        input_length = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )

        return response.strip()


# ==========================================
if __name__ == "__main__":
    generator = AnswerGenerator()

    fake_question = (
        "How to configure OpenAI compatible server?"
    )
    fake_chunks: List[Dict[str, Any]] = [
        {
            "text": (
                "To configure the OpenAI compatible"
                " server in vLLM, you need to use"
                " the command: python -m"
                " vllm.entrypoints.openai.api_server"
                " --model facebook/opt-125m."
            ),
        },
        {
            "text": (
                "OpenAI you need to use the command"
            ),
        },
    ]

    print("\n L'Model kayfker...")
    answer = generator.generate(
        fake_question, fake_chunks
    )
    print(f"\nwen Jawb:\n{answer}")
