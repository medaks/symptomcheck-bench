from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from medask.models.orm.models import Lang, Role


class CMessage(BaseModel):
    user_id: int
    role: Role
    body: str
    id: Optional[int] = None  # Populated by the server.
    chat_id: Optional[int] = None  # None when a new chat is started.
    # Set to True when body contains the end (fin) of a diagnosis. In that case, body will be
    # jsonified list of diagnoses - dicts with 3 keys: 'icd', 'disease', 'reasoning'.
    fin: bool = False
    # Used only in /diagnosis endpoint. Set to, for example Lang.DUTCH, means self.body is
    # in Dutch and response to this message will come in Dutch.
    lang: Lang = Lang.UNKNOWN
    # Only used in /diagnosis endpoint. LLM reasoning explaining why it posed a question.
    explanation: Optional[str] = None

    def to_openai(self) -> Dict[str, str]:
        """Convert to message expected by OpenAI API."""
        return {"role": self.role.value.lower(), "content": self.body}

    def to_anthropic(self) -> Dict[str, str]:
        """Convert to message expected by Anthropic API."""
        return {"role": self.role.value.lower(), "content": self.body}

    def __str__(self) -> str:
        out = f"<CMessage id: {self.id}> {self.lang.name}\n"
        out += f"user_id: {self.user_id}, chat_id: {self.chat_id}\n"
        out += f"{self.role.value}: |{self.body}|\n"
        out += f"</CMessage id: {self.id}>\n"
        return out

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def esl(self) -> bool:
        """True if self not in english."""
        return self.lang not in [Lang.UNKNOWN, Lang.ENGLISH]


class CChat(BaseModel):
    user_id: int
    messages: List[CMessage] = []
    id: Optional[int] = None

    def model_post_init(self, __context: Any) -> None:
        """Check id and user_id match with messages."""
        for msg in self.messages:
            assert msg.chat_id == self.id, f"Chat id must match: {msg} {self.id}"
            assert msg.user_id == self.user_id, f"User id must match: {msg} {self.user_id}"

    def __str__(self) -> str:
        out = f"<CChat id: {self.id}>\n"
        out += f"user_id: {self.user_id}\n"
        for i, msg in enumerate(self.messages):
            out += f"\nmsg {i}"
            out += f"\n{msg}"
        out += f"</CChat id: {self.id}>\n\n"
        return out

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.messages)
