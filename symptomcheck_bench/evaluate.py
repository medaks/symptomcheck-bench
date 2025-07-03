from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict

from medask.models.comms.models import CMessage
from medask.models.orm.models import Role
from medask.ummon.openai import UmmonOpenAI

from medask.benchmark.simulator import NaiveSimulator

if TYPE_CHECKING:
    from medask.benchmark.experiment_result import ExperimentResult

logger = getLogger("benchmark.evaluate")
_ummon_openai = UmmonOpenAI("gpt-4o")


def _get_score(obtained_diagnoses: str, correct_diagnosis: str) -> int:
    body = f"""Given a list of differential diagnoses and the correct diagnosis. Determine if any diagnosis in the list is either
        an exact match or extremely relevant to the correct diagnosis.     A diagnosis is considered extremely relevant if it is:
        1. A direct subtype/variant of the condition (e.g., "Alzheimer's Disease" matches "Dementia")
        2. A broader category that includes the condition (e.g., "Head Injury" matches "Concussion")
        3. A temporal variation (e.g., "Acute Bronchitis" matches "Chronic Bronchitis")
        4. A closely related condition with shared pathophysiology and clinical presentation (e.g., "Gout" matches "Pseudogout")
        5. A condition with established pathophysiological link (e.g., "Chronic Sinusitis" matches "Nasal Polyps")
            
        If any diagnosis meets these criteria, specify its position, starting from 1.
        If none of the diagnoses meet these criteria, write -1.
        Respond in the following format: Correct diagnosis position: [number]

        OBTAINED DIAGNOSES: {obtained_diagnoses}
        CORRECT DIAGNOSIS: {correct_diagnosis}
    """
    cmsg = CMessage(user_id=1, body=body, role=Role.SYSTEM)
    out = _ummon_openai.inquire(cmsg).body
    try:
        # Extract just the number from "Correct diagnosis position: [number]"
        position_part = out.split("Position:")[1].strip()
        position_str = position_part.replace('[', '').replace(']', '').strip()
        position = int(position_str)
        return position
    except Exception:
        logger.exception(f"FAILED: {obtained_diagnoses=} {correct_diagnosis=} {out=}")
        return -3


def get_score(obtained_diagnoses: str, correct_diagnosis: str) -> float:
    position = _get_score(obtained_diagnoses, correct_diagnosis)
    print(f"position={position}\t{correct_diagnosis}\t{obtained_diagnoses}")
    return float(position)

def evaluate(result: "ExperimentResult") -> Dict[int, Dict[str, Any]]:
    """
    Evaluation.
    Super hacky for now, we need to reconstruct the right Simulator, to know if the
    chat successfully finish and to extract the diagnosis.
    :return: For each experiment from <num_experiments>, a dict of experiment results.
    """
    result = result.copy()  # Make sure we don't accidentally modify the results.
    results: Dict[int, Dict[str, Any]] = {i: {} for i in range(result.num_experiments)}
    
    for i in range(result.num_experiments):
        positions = []
        chats = result.chats[i]
        for chat, vignette in zip(chats, result.vignettes):
            simulator = NaiveSimulator(vignette, None, None)
            simulator.chat_doctor = chat
            if not simulator.diagnosis_finished:
                logger.warning("Simulation did not finish with a diagnosis.")
                positions.append(-2)
                continue

            obtained_diagnoses = simulator.extract_diagnoses()
            correct_diagnosis = vignette.correct_diagnosis
            positions.append(get_score(obtained_diagnoses, correct_diagnosis))

        logger.info(f"Results of run {i=}")
        goods = [p for p in positions if p >= 1]  # Positions of correct diagnoses.
        avg_position = sum(goods) / len(goods) if goods else -1
        print(f"\tpositions={positions}")
        print(f"\tNumber of correct diagnoses: {len(goods)} / {len(positions)}")
        print(f"\tAverage position of correct diagnosis: {avg_position}")
        print("\n\n")
        results[i]["n_correct"] = len(goods)
        results[i]["positions"] = positions

    return results
