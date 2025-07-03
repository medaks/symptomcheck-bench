from logging import getLogger
from time import sleep
from typing import Dict, List, Optional

from anthropic import Anthropic, RateLimitError

from medask.const import KEY_ANTHROPIC
from medask.models.comms.models import CMessage
from medask.models.orm.models import Role
from medask.util.decorator import timeit
from medask.util.gen_cmsg import gen_cmsg
from medask.ummon.base import BaseUmmon

client = Anthropic(api_key=KEY_ANTHROPIC)
logger = getLogger("ummon.anthropic")


class UmmonAnthropic(BaseUmmon):
    def __init__(self, model: Optional[str] = None) -> None:
        self._model = model or "claude-3-haiku-20240307"

    def _converse_raw(self, history: List[Dict[str, str]]) -> str:
        params = dict(
            model=self._model,
            messages=history,
            max_tokens=600,
        )

        # Anthropic takes SYSTEM prompts in a separate field.
        if len(history) > 1 and history[0]["role"] == "system":
            params["system"] = history[0]["content"]
            params["messages"] = history[1:]

        slep = 3.0
        while slep < 60:
            try:
                out = client.messages.create(**params)
                if out.stop_reason == "max_tokens":
                    logger.warning(f"Max tokens reached at {out}")
                return out.content[0].text
            except RateLimitError:
                logger.info(f"Rate limit, sleeping for {round(slep, 1)} seconds.")
                sleep(slep)
                slep *= 1.5

        raise RuntimeError("Too much rate limiting")

    @timeit(logger, log_kwargs=False)
    def inquire(self, prompt: CMessage, json: bool = False) -> CMessage:
        prompt_raw = prompt.to_anthropic()
        retort: str = self._converse_raw([prompt_raw])
        return gen_cmsg(prompt, body=retort, role=Role.ASSISTANT)

    @timeit(logger, log_kwargs=False)
    def converse(self, history: List[CMessage]) -> CMessage:
        history_raw = [msg.to_anthropic() for msg in history]
        retort: str = self._converse_raw(history_raw)

        return gen_cmsg(history[-1], body=retort, role=Role.ASSISTANT)
