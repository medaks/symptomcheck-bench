import logging
from argparse import ArgumentParser
from random import sample
from typing import TYPE_CHECKING, List

from medask.ummon.anthropic import UmmonAnthropic
from medask.ummon.local_llm import UmmonLocalLLM
from medask.ummon.koboldcpp import UmmonKoboldCPP
from medask.util.concurrency import exec_concurrently
from medask.util.decorator import timeit
from medask.util.log import get_logger

from medask.benchmark.evaluate import evaluate
from medask.benchmark.experiment_result import ExperimentResult
from medask.benchmark.simulator import LocalSimulator, NaiveSimulator
from medask.benchmark.util import LLMClient, model_to_client
from medask.benchmark.vignette import (
    Vignette,
    load_vignettes,
)

if TYPE_CHECKING:
    from medask.benchmark.simulator import Simulator

logger = get_logger("benchmark")
logging.getLogger("ummon.anthropic").setLevel(logging.WARNING)
logging.getLogger("ummon.koboldcpp").setLevel(logging.WARNING)
logging.getLogger("ummon.mistral").setLevel(logging.WARNING)
logging.getLogger("ummon.openai").setLevel(logging.WARNING)


@timeit(logger, log_kwargs=False)
def run_experiment(
    vignettes: List["Vignette"], doctor_client: LLMClient, patient_client: LLMClient
) -> List["Simulator"]:
    """
    Make a Simulator object for each vignette and use them to simulate the diagnoses.
    Execute them concurrently for speedup.
    """
    simulators = []
    if isinstance(doctor_client, UmmonLocalLLM):
        simulator_cls = LocalSimulator
    else:
        simulator_cls = NaiveSimulator

    # Initialise simulator with a vignette and new instances of clients.
    for v in vignettes:
        simulator = simulator_cls(
            vignette=v,
            doctor_client=type(doctor_client)(model=doctor_client._model),
            patient_client=type(patient_client)(model=patient_client._model),
        )
        simulators.append(simulator)

    # Some clients cannot be run concurrently because of rate limiting.
    max_workers = 10
    for client in (doctor_client, patient_client):
        if isinstance(client, (UmmonAnthropic)):
            max_workers = 2
        elif isinstance(client, (UmmonLocalLLM, UmmonKoboldCPP)):
            max_workers = 1

    # Concurrently call .simulate() on each of the simulators.
    params = [{} for _ in simulators]
    exec_concurrently([s.simulate for s in simulators], params, max_workers)

    # Return simulators, which contain chats in attributes (self.chat_doctor).
    return simulators


def get_args() -> ArgumentParser:
    parser = ArgumentParser(description="Symptom Assessment Simulation")
    models = "gpt-4o, claude-3-haiku-20240307, open-mixtral-8x7b ..."
    parser.add_argument("--doctor_llm", type=str, default="gpt-4o-mini", help=models)
    parser.add_argument("--patient_llm", type=str, default="gpt-4o-mini")
    parser.add_argument("--evaluator_llm", type=str, default="gpt-4o")
    parser.add_argument(
        "--file",
        type=str,
        choices=["avey"],
        required=True,
        help="Specify 'avey' for AVEY_VIGNETTES (more vignettes choices to come later)",
    )
    parser.add_argument(
        "--num_vignettes", type=int, default=10, help="Number of vignettes used in the experiment."
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=1,
        help="Number of iterations through the vignettes (default: 1)",
    )
    parser.add_argument(
        "--comment", type=str, help="Optional comment to include in the experiment result."
    )
    parser.add_argument(
        "--result_name_suffix",
        type=str,
        default="",
        help="Optional suffex to add to the filename with the experiment result.",
    )

    return parser


def main(args: ArgumentParser) -> None:
    args = args.parse_args()

    # Get random sample of <num_vignettes> from the right vignette file.
    vignettes = load_vignettes(args.file)
    num_vignettes = min(args.num_vignettes, len(vignettes))
    indices = sorted(sample(list(range(len(vignettes))), num_vignettes))
    logger.info(f"Running experiment over vignettes {indices}")
    vignettes = [v for i, v in enumerate(vignettes) if i in indices]

    # Instantiate correct API clients.
    doctor_client = model_to_client(args.doctor_llm)
    patient_client = model_to_client(args.patient_llm)

    # Create result file.
    result = ExperimentResult(
        vignette_file=args.file,
        vignettes=vignettes,
        vignette_indices=indices,
        num_experiments=args.num_experiments,
        doctor_llm=args.doctor_llm,
        patient_llm=args.patient_llm,
        chats=[],
        comment=args.comment,
        result_name_suffix=args.result_name_suffix,
    )

    # Run experiment
    for _ in range(args.num_experiments):
        simulators = run_experiment(vignettes, doctor_client, patient_client)
        result.chats.append([s.chat_doctor for s in simulators])

        # Do dump of current results, overwriting at each step.
        # So partial results are stored even if an error occurs midway in the experiment.
        logger.info(f"Dumping results to {result.dump_path}")
        result.dump()

    # Run evaluation
    result.evaluation = evaluate(result)
    result.dump()


if __name__ == "__main__":
    args = get_args()
    main(args)
