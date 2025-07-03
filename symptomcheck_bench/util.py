from typing import TypeVar

from medask.ummon.anthropic import UmmonAnthropic
from medask.ummon.deepseek import UmmonDeepSeek
from medask.ummon.koboldcpp import UmmonKoboldCPP
from medask.ummon.local_llm import UmmonLocalLLM
from medask.ummon.mistral import UmmonMistral
from medask.ummon.openai import UmmonOpenAI

# My autismo, type created to created all LLM clients used in the benchmark.
LLMClient = TypeVar("LLMClient", UmmonOpenAI, UmmonAnthropic, UmmonMistral)


def model_to_client(model: str) -> LLMClient:  # type: ignore
    if model == "medask-local":
        raise RuntimeError("not yet public")
    elif model == "medask":
        raise RuntimeError("not yet public")
    elif "gpt" in model:
        return UmmonOpenAI(model)
    elif "claude" in model:
        return UmmonAnthropic(model)
    elif "mistral" in model or "mixtral" in model:
        return UmmonMistral(model)
    elif "koboldcpp+http://" in model:
        return UmmonKoboldCPP(model[len("koboldcpp+") :])
    elif "http://" in model:
        return UmmonLocalLLM(model)
    elif "deepseek" in model:
        return UmmonDeepSeek()
    else:
        raise ValueError(f"Unsupported model {model}")
