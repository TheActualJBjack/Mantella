from src.config.config_loader import ConfigLoader
from src.llm.openai_client import OpenAIClient
from src.llm.gemini_client import GeminiClient
from src.llm.client_base import ClientBase

def LlmClientFactory(config: ConfigLoader, secret_key_file: str, image_secret_key_file: str) -> ClientBase:
    if config.llm_api.lower() == "gemini":
        api_key = ClientBase._get_api_key([secret_key_file])
        return GeminiClient(
            api_key=api_key,
            llm=config.llm,
            llm_params=config.llm_params,
            custom_token_count=config.custom_token_count,
            secret_key_files=[secret_key_file]
        )
    else:
        return OpenAIClient(
            api_url=config.llm_api,
            llm=config.llm,
            llm_params=config.llm_params,
            custom_token_count=config.custom_token_count,
            secret_key_files=[secret_key_file]
        )