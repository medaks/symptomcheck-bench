from abc import abstractmethod, ABC
from typing import List

from medask.models.comms.models import CMessage


class BaseUmmon(ABC):
    @abstractmethod
    def inquire(self, prompt: CMessage) -> CMessage:
        pass

    @abstractmethod
    def converse(self, history: List[CMessage]) -> CMessage:
        pass
