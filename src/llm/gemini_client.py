import os
from google import genai
from google.genai import types
from src.llm.client_base import ClientBase

from src.llm.message_thread import message_thread
from src.llm.messages import Message
import logging

class GeminiClient(ClientBase):
    def __init__(self, api_key: str, llm: str, llm_params: dict, custom_token_count: int, secret_key_files: list[str]) -> None:
        super().__init__("Gemini", llm, llm_params, custom_token_count, secret_key_files)
        # Configure the Gemini client
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", api_key))
        self.model = genai.GenerativeModel(self.model_name)

    async def streaming_call(self, messages: message_thread | Message, is_multi_npc: bool):
        with self._generation_lock:
            logging.log(28, 'Getting LLM response...')

            if isinstance(messages, Message):
                gemini_messages = [messages.get_gemini_message()]
            else:
                gemini_messages = messages.get_gemini_messages()

            config = self._request_params or {}

            try:
                response = self.model.generate_content(gemini_messages, stream=True, generation_config=config)
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            except Exception as e:
                logging.error(f"Gemini API Error: {e}")
                yield None

    def request_call(self, messages: message_thread | Message):
        with self._generation_lock:
            logging.log(28, 'Getting LLM response...')

            if isinstance(messages, Message):
                gemini_messages = [messages.get_gemini_message()]
            else:
                gemini_messages = messages.get_gemini_messages()

            config = self._request_params or {}

            try:
                response = self.model.generate_content(gemini_messages, generation_config=config)
                return response.text
            except Exception as e:
                logging.error(f"Gemini API Error: {e}")
                return None
