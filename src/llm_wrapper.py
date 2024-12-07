from groq import Groq


class LLMWrapper:
    """
    Wrapper for the LLM API using Groq.
    """

    def __init__(self, api_key: str, model_name: str = "llama3-8b-8192"):
        """
        Initialize the LLMWrapper and set the API key.

        :param api_key: The API key for the LLM.
        :param model_name: The name of the LLM model.
        """
        self.model_name = model_name
        self.model = Groq(api_key=api_key)
        self.system_prompt = """
            You are a Witcher expert with deep knowledge of first books.
            Your task is to answer the question based only on the provided context.
            Carefully consider all the information, and provide a concise and accurate answer.
            ALWAYS cite the sources from context as numbers in square brackets. For example: [1].
        """

    def generate_answer(self, prompt: str) -> str:
        """
        Generate an answer using the LLM.

        :param prompt: The prompt to send to the LLM.
        :return: The generated answer.
        """
        # generate llm answer
        chat_completion = self.model.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=1,
            model=self.model_name,
        )

        return chat_completion.choices[0].message.content
