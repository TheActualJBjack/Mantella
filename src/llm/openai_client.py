import logging
import time
from typing import AsyncGenerator, Any
from openai import APIConnectionError, BadRequestError, AsyncOpenAI, OpenAI
from src.llm.client_base import ClientBase
from src.llm.message_thread import message_thread
from src.llm.messages import Message, ImageMessage, UserMessage
import src.utils as utils

class OpenAIClient(ClientBase):
    def __init__(self, api_url: str, llm: str, llm_params: dict, custom_token_count: int, secret_key_files: list[str]) -> None:
        super().__init__(api_url, llm, llm_params, custom_token_count, secret_key_files)
        self._startup_async_client: AsyncOpenAI | None = self.generate_async_client() # initialize first client in advance of sending first LLM request to save time

    @utils.time_it
    def generate_async_client(self) -> AsyncOpenAI:
        """Generates a new AsyncOpenAI client already setup to be used right away.
        Close the client after usage using 'await client.close()'

        The client needs to be closed after every call (and a new one created for the next call) to avoid connection issues

        Use :func:`streaming_call` for a normal streaming call to the LLM

        Returns:
            AsyncOpenAI: The new async client object
        """
        return AsyncOpenAI(api_key=self._api_key, base_url=self._base_url, default_headers=self._header)


    @utils.time_it
    def generate_sync_client(self) -> OpenAI:
        """Generates a new OpenAI client already setup to be used right away.
        Close the client after usage using 'client.close()'

        Use :func:`request_call` for a normal call to the LLM

        Returns:
            OpenAI: The new sync client object
        """
        return OpenAI(api_key=self._api_key, base_url=self._base_url, default_headers=self._header)

    @utils.time_it
    def request_call(self, messages: Message | message_thread) -> str | None:
        with self._generation_lock:
            sync_client = self.generate_sync_client()
            chat_completion = None
            logging.log(28, 'Getting LLM response...')

            if isinstance(messages, Message) or isinstance(messages, ImageMessage):
                openai_messages = [messages.get_openai_message()]
            else:
                openai_messages = messages.get_openai_messages()

            if self._request_params:
                request_params = self._request_params
            else:
                request_params: dict[str, Any] = {}
            try:
                chat_completion = sync_client.chat.completions.create(
                    model=self.model_name,
                    messages=openai_messages,
                    **request_params,
                )
            except RateLimitError:
                logging.warning('Could not connect to LLM API, retrying in 5 seconds...')
                time.sleep(5)
            finally:
                sync_client.close()

            if (
                not chat_completion or
                not chat_completion.choices or
                chat_completion.choices.__len__() < 1 or
                not chat_completion.choices[0].message.content
            ):
                logging.info(f"LLM Response failed")
                return None

            reply = chat_completion.choices[0].message.content
            return reply


    @utils.time_it
    async def streaming_call(self, messages: Message | message_thread, is_multi_npc: bool) -> AsyncGenerator[str | None, None]:
        with self._generation_lock:
            logging.log(28, 'Getting LLM response...')

            if self._startup_async_client:
                async_client = self._startup_async_client
                self._startup_async_client = None # do not reuse the same client
            else:
                async_client = self.generate_async_client()

            if self._request_params:
                request_params = self._request_params.copy() # copy of self._request_params to allow temporary override
            else:
                request_params: dict[str, Any] = {}
            if is_multi_npc: # override max_tokens to be at least 250 in radiant / multi-NPC conversations
                request_params["max_tokens"] = max(self.max_tokens_param, 250)
            try:
                # Prepare the messages including the image if provided
                vision_hints = ''
                if isinstance(messages, Message):
                    openai_messages = [messages.get_openai_message()]
                    if isinstance(messages, UserMessage):
                        vision_hints = messages.get_ingame_events_text()
                else:
                    openai_messages = messages.get_openai_messages()
                    last_message = messages.get_last_message()
                    if isinstance(last_message, UserMessage):
                        vision_hints = last_message.get_ingame_events_text()
                if self._image_client:
                    openai_messages = self._image_client.add_image_to_messages(openai_messages, vision_hints)

                async for chunk in await async_client.chat.completions.create(
                    model=self.model_name,
                    messages=openai_messages,
                    stream=True,
                    **request_params,
                ):
                    try:
                        if chunk and chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                    except Exception as e:
                        logging.error(f"LLM API Connection Error: {e}")
                        break
            except Exception as e:
                utils.play_error_sound()
                if isinstance(e, APIConnectionError):
                    if e.code in [401, 'invalid_api_key']: # incorrect API key
                        if self._base_url == 'https://api.openai.com/v1':
                            service_connection_attempt = 'OpenRouter' # check if player means to connect to OpenRouter
                        else:
                            service_connection_attempt = 'OpenAI' # check if player means to connect to OpenAI
                        logging.error(f"Invalid API key. If you are trying to connect to {service_connection_attempt}, please choose an {service_connection_attempt} model via the 'model' setting in MantellaSoftware/config.ini. If you are instead trying to connect to a local model, please ensure the service is running.")
                    else:
                        logging.error(f"LLM API Error: {e}")
                elif isinstance(e, BadRequestError):
                    if (e.type == 'invalid_request_error') and (self._image_client): # invalid request
                        logging.error(f"Invalid request. Try disabling Vision in Mantella's settings and try again.")
                    else:
                        logging.error(f"LLM API Error: {e}")
                else:
                    logging.error(f"LLM API Error: {e}")
            finally:
                await async_client.close()
