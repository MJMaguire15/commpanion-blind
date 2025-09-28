import lmstudio as lms

class LMStudioResponder:
    """
    A wrapper around the LM Studio chat model to generate streaming responses.
    """

    def __init__(self, model_name: str = None, system_prompt: str = None):
        """
        Initialize the LMStudioResponder.

        Args:
            model_name (str | None): Optional name of the model to use. Default: first llm load on the server
            system_prompt (str | None): Optional custom system prompt. Default: You are a friendly assistant.
        """
        self.model = lms.llm(model_name) if model_name else lms.llm()
        self.chat = lms.Chat(
            system_prompt or "You are a vocal assistant. Answer concisely with 1 or 2 sentences. Never use emojis."
        )

    def respond(self, prompt: str):
        """
        Generate a streaming response from the model.

        Args:
            prompt (str): The user prompt to respond to.

        Returns:
            Generator yielding response chunks.
        """
        self.chat.add_user_message(prompt)
        return self.model.respond_stream(
            self.chat,
            on_message=self.chat.append,
        )