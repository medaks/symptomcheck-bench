from typing import Any

from medask.models.comms.models import CMessage


def gen_cmsg(template: CMessage, **kwargs: Any) -> CMessage:
    """Return a copy of <template> whose attributes got overwriten by <kwargs>."""
    cmsg_dict = template.model_dump()
    cmsg_dict.pop("id")  # Remove id.
    cmsg_dict.update(**kwargs)  # Potentially overwrite <template> attributes.
    return CMessage.model_validate(cmsg_dict)
