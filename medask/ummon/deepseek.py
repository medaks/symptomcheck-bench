from logging import getLogger
from time import sleep
from typing import Dict, List, Optional

from openai import OpenAI, RateLimitError

from medask.const import KEY_DEEPSEEK
from medask.models.comms.models import CMessage
from medask.models.orm.models import Role
from medask.util.decorator import timeit
from medask.util.gen_cmsg import gen_cmsg
from medask.ummon.base import BaseUmmon

logger = getLogger("ummon.deepseek")
client = OpenAI(api_key=KEY_DEEPSEEK, timeout=60, base_url="https://api.deepseek.com")


class UmmonDeepSeek(BaseUmmon):
    def __init__(self, model: Optional[str] = None) -> None:
        self._model = model or "deepseek-chat"

    def _converse_raw(self, history: List[Dict[str, str]], json: bool) -> str:
        params = dict(
            model=self._model,
            messages=history,
        )
        if json:
            params["response_format"] = {"type": "json_object"}

        slep = 3.0
        while slep < 60:
            try:
                completion = client.chat.completions.create(**params)
                return completion.choices[0].message.content
            except RateLimitError:
                logger.info(f"Rate limit, sleeping for {round(slep, 1)} seconds.")
                sleep(slep)
                slep *= 1.5

        raise RuntimeError("Too much rate limiting")

    @timeit(logger, log_kwargs=False)
    def inquire(self, prompt: CMessage, json: bool = False) -> CMessage:
        prompt_raw = prompt.to_openai()
        retort: str = self._converse_raw([prompt_raw], json=json)
        return gen_cmsg(prompt, body=retort, role=Role.ASSISTANT)

    @timeit(logger, log_kwargs=False)
    def converse(self, history: List[CMessage], json: bool = False) -> CMessage:
        history_raw = [msg.to_openai() for msg in history]
        retort: str = self._converse_raw(history_raw, json=json)

        return gen_cmsg(history[-1], body=retort, role=Role.ASSISTANT)