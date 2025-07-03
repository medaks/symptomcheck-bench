import json
from typing import Dict, List

from medask.models.comms.models import CMessage
from medask.models.orm.models import Role
from medask.util.client import post
from medask.util.decorator import timeit
from medask.util.log import get_logger
from medask.ummon.base import BaseUmmon

logger = get_logger("UmmonLocalLLM")


class UmmonLocalLLM(BaseUmmon):
    def __init__(self, model: str) -> None:
        """Note, model is actually an url to the server."""
        # Example url: http://localhost:5013.
        assert "http" in model, f"param model should point to the server, not {model}"
        self._url = model
        self._model = model  # hack for benchmark.

    def _converse_raw(self, history: List[Dict[str, str]]) -> str:
        assert len(history) == 1, "Other things unsupported for now"
        body = json.dumps(history[0])
        resp = post("inquire", body=body, url=self._url)
        return resp

    def _raw_to_out(self, user_id: int, chat_id: int, raw: str) -> CMessage:
        return CMessage(
            user_id=user_id,
            chat_id=chat_id,
            role=Role.ASSISTANT,
            body=raw,
        )

    @timeit(logger, log_kwargs=False)
    def inquire(self, prompt: CMessage) -> CMessage:
        prompt_raw = prompt.to_openai()
        retort: str = self._converse_raw([prompt_raw])
        return self._raw_to_out(prompt.user_id, prompt.chat_id, retort)

    @timeit(logger, log_kwargs=False)
    def converse(self, history: List[CMessage]) -> CMessage:
        history_raw = [msg.to_openai() for msg in history]
        retort: str = self._converse_raw(history_raw)

        msg = history[-1]
        return self._raw_to_out(msg.user_id, msg.chat_id, retort)
