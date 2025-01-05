import logging
import traceback
from flask import request
from datetime import datetime
from requests import get, post
from json import dumps, loads
import os

from server.config import special_instructions

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Backend_Api:
    def __init__(self, app, config: dict) -> None:
        self.app = app
        self.openai_key = os.getenv("OPENAI_API_KEY") or config['openai_key']
        self.openai_api_base = os.getenv("OPENAI_API_BASE") or config['openai_api_base']
        self.proxy = config.get('proxy', {})
        self.routes = {
            '/backend-api/v2/conversation': {
                'function': self._conversation,
                'methods': ['POST']
            }
        }

    def _conversation(self):
        try:
            # Проверка входных данных
            if not request.json:
                return {'success': False, 'message': 'Request body is missing'}, 400

            required_keys = ['jailbreak', 'meta']
            missing_keys = [key for key in required_keys if key not in request.json]
            if missing_keys:
                return {
                    'success': False,
                    'message': f'Missing required keys: {", ".join(missing_keys)}'
                }, 400

            jailbreak = request.json.get('jailbreak', 'default')
            meta = request.json.get('meta', {}).get('content', {})
            internet_access = meta.get('internet_access', False)
            _conversation = meta.get('conversation', [])
            prompt = meta.get('parts', [{}])[0]

            current_date = datetime.now().strftime("%Y-%m-%d")
            system_message = f'You are ChatGPT also known as ChatGPT, a large language model trained by OpenAI. Strictly follow the users instructions. Knowledge cutoff: 2021-09-01 Current date: {current_date}'

            # Подготовка дополнительного контекста
            extra = []
            if internet_access:
                search_results = self._web_search(prompt.get("content", ""))
                extra = [{'role': 'user', 'content': search_results}]

            # Составление полного разговора
            conversation = [{'role': 'system', 'content': system_message}] + \
                           extra + special_instructions.get(jailbreak, []) + \
                           _conversation + [prompt]

            # Отправка запроса в OpenAI API
            url = f"{self.openai_api_base}/v1/chat/completions"
            proxies = self._get_proxies()

            logger.info(f"Sending request to OpenAI API: {url}")
            gpt_resp = post(
                url=url,
                proxies=proxies,
                headers={'Authorization': f'Bearer {self.openai_key}'},
                json={
                    'model': request.json.get('model'),
                    'messages': conversation,
                    'stream': True
                },
                stream=True,
                timeout=30  # Таймаут для предотвращения зависания
            )

            # Обработка ответа
            if gpt_resp.status_code >= 400:
                error_data = self._handle_error_response(gpt_resp)
                return error_data, gpt_resp.status_code

            # Потоковая передача ответа
            return self.app.response_class(self._stream_response(gpt_resp), mimetype='text/event-stream')

        except Exception as e:
            logger.error(f"Error in _conversation: {str(e)}", exc_info=True)
            return {
                '_action': '_ask',
                'success': False,
                "error": f"An error occurred: {str(e)}",
                "details": traceback.format_exc()
            }, 400

    def _web_search(self, query):
        try:
            search = get('https://ddg-api.herokuapp.com/search', params={
                'query': query,
                'limit': 3,
            }, timeout=10)
            search.raise_for_status()

            blob = ''
            for index, result in enumerate(search.json()):
                blob += f'[{index}] "{result.get("snippet", "")}"\nURL:{result.get("link", "")}\n\n'

            date = datetime.now().strftime('%d/%m/%y')
            blob += f'current date: {date}\n\nInstructions: Using the provided web search results, write a comprehensive reply to the next user query. Make sure to cite results using [[number](URL)] notation after the reference.'

            return blob
        except Exception as e:
            logger.error(f"Web search error: {str(e)}", exc_info=True)
            return "Error: Unable to retrieve search results."

    def _get_proxies(self):
        if self.proxy.get('enable', False):
            return {
                'http': self.proxy.get('http'),
                'https': self.proxy.get('https'),
            }
        return None

    def _handle_error_response(self, gpt_resp):
        try:
            error_data = gpt_resp.json().get('error', {})
        except ValueError:
            error_data = {"message": "Invalid JSON response from OpenAI API"}
        logger.error(f"OpenAI API error: {error_data}")
        return {
            'success': False,
            'error_code': error_data.get('code'),
            'message': error_data.get('message', 'An error occurred'),
            'status_code': gpt_resp.status_code
        }

    def _stream_response(self, gpt_resp):
        for chunk in gpt_resp.iter_lines():
            try:
                decoded_line = loads(chunk.decode("utf-8").split("data: ")[1])
                token = decoded_line["choices"][0]['delta'].get('content')
                if token:
                    yield token
            except Exception as e:
                logger.error(f"Stream error: {str(e)}", exc_info=True)
                continue
