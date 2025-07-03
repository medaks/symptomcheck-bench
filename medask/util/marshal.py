from typing import TYPE_CHECKING, List, Tuple

from medask.models.orm.models import Role

if TYPE_CHECKING:
    from medask.models.comms.models import CMessage


def _is_diagnosis(messages: List["CMessage"]) -> bool:
    if messages == []:
        return False
    msg = messages[0]
    return msg.role == Role.SYSTEM and "Pretend you're a doctor" in msg.body


def marshal(
    messages: List["CMessage"], rename_roles: bool = False
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Ignore all SYSTEM msgs.
    Starting from first USER message:
        if current msg role == ASSISTANT:
            add marshalled partial convo to current msg id
        Add msg to marshalled partial convo
    : Returns: Marshalled convo from first USER to end, and tuples to index
        tuple to index: (<partial convo from USER to USER>, <body of ASSISTANT reply>)
    """
    msgs = [m for m in messages if m.role != Role.SYSTEM]

    to_index = []
    marshalled: str = ""
    for i, msg in enumerate(msgs):
        if msg.role == Role.ASSISTANT and marshalled != "":
            assert (
                msgs[i - 1].role == Role.USER
            ), f"USER not prev of msg {msg.id} in chat {msg.chat_id}"
            to_index.append((marshalled, msg.body))
            marshalled += f"{msg.role.value}: {msg.body}\n\n"
        elif msg.role == Role.USER:
            marshalled += f"{msg.role.value}: {msg.body}\n\n"

    if rename_roles:
        marshalled = marshalled.replace("USER", "PATIENT").replace("ASSISTANT", "DOCTOR")
    return marshalled, to_index
