from openai import OpenAI

from deepscholar.config import settings


class LLMClient:

    def __init__(self):
        self.client = OpenAI(
            api_key=(
                settings.llm_api_key
                .get_secret_value()
            ),
            base_url=settings.llm_base_url,
        )
        self.model = settings.llm_model

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0,
        max_tokens: int = 1000
    ) -> str:
        response = (
            self.client
            .chat
            .completions
            .create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                },
            )
        )
        content = (
            response
            .choices[0]
            .message
            .content
        )
        if not content:
            raise RuntimeError(
                "LLM returned empty content."
            )

        return content