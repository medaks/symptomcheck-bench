import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from medask.models.comms.models import CChat

from medask.benchmark.vignette import (
    AveyVignette,
    Vignette,
)


class ExperimentResult(BaseModel):
    """
    :param vignettes: Vignettes from <vignette_file> that were used in this experiment.
    :param vignette_indices: An experiment is usually run over a random subsample of all
        the vignettes in <vignette_file>. The indices of the subsample are stored, to make
        it easier to reproduce an experiment.
    :param chats: One experiment generates a list of doctor chats, one chat for each vignette.
        There are as many lists as <num_experiments>.
    :param result_name_suffix: Add a suffix to the filename where this result is stored.
    :param evaluation: Stores result of benchmark.evaluate. This is just a simple dict,
        so it will always be backward compatible.
    """

    vignette_file: str
    vignettes: List[Vignette]
    vignette_indices: List[int]
    num_experiments: int
    doctor_llm: str
    patient_llm: str
    chats: List[List[CChat]]
    dt: datetime = datetime.now()
    comment: Optional[str] = None
    result_name_suffix: str = ""
    evaluation: Dict[Any, Any] = {}

    @property
    def dump_path(self) -> str:
        dt = self.dt.isoformat(timespec="seconds")
        suffix = f"_{self.result_name_suffix}" if self.result_name_suffix else ""
        name = f"{dt}_{self.doctor_llm}_{len(self.chats[0])}{suffix}.json"
        if "http" in self.doctor_llm:
            name = name.replace(self.doctor_llm, "LOCAL_LLM")
        directory = os.path.dirname(os.path.abspath(__file__))
        return f"{directory}/results/{name}"

    def dump(self) -> None:
        with open(self.dump_path, "w") as f:
            f.write(self.model_dump_json())

    @classmethod
    def load(cls, path: str) -> "ExperimentResult":
        with open(path) as f:
            raw = json.load(f)
            # Transform str indices into ints, like '0' into 0.
            for exp_ix in [k for k in raw["evaluation"]]:
                evaluation = raw["evaluation"].pop(exp_ix)
                raw["evaluation"][int(exp_ix)] = evaluation
            cls = AveyVignette if raw["vignette_file"] == "avey" else None
            raw["vignettes"] = [cls(**d) for d in raw["vignettes"]]
        return ExperimentResult(**raw)
