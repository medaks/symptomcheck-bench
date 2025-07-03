import json
from abc import abstractmethod
from logging import getLogger

from medask.models.comms.models import CChat, CMessage
from medask.models.orm.models import Role
from medask.util.decorator import timeit
from medask.util.gen_cmsg import gen_cmsg
from medask.util.marshal import marshal

from medask.benchmark.agent import Doctor, Patient
from medask.benchmark.util import LLMClient
from medask.benchmark.vignette import Vignette

logger = getLogger("benchmark.simulator")


class Simulator:
    def __init__(
        self, vignette: "Vignette", doctor_client: "LLMClient", patient_client: "LLMClient"
    ) -> None:
        self.vignette = vignette
        self.doctor_client = doctor_client
        self.patient_client = patient_client
        self.doctor = Doctor(vignette)
        patient = Patient(vignette)
        self.max_len = 24
        user_id = 5
        self.chat_doctor = CChat(
            user_id=user_id,
            messages=[
                CMessage(user_id=user_id, role=Role.SYSTEM, body=self.doctor.system_prompt()),
            ],
        )
        self.chat_patient = CChat(
            user_id=user_id,
            messages=[
                CMessage(user_id=user_id, role=Role.SYSTEM, body=patient.system_prompt()),
                CMessage(user_id=user_id, role=Role.USER, body=self.doctor.initial_prompt()),
            ],
        )

    @abstractmethod
    def infer_doctor(self) -> CMessage:
        """
        Generate a new doctor message using self.chat_doctor.
        Does not modify any attributes.
        """
        pass

    @abstractmethod
    def infer_patient(self) -> CMessage:
        """
        Generate a new patient message using self.chat_patient.
        Does not modify any attributes.
        """
        pass

    @property
    @abstractmethod
    def diagnosis_finished(self) -> bool:
        """
        Return True if the chat obtained from self.simulate() successfully finished
        by providing diagnoses.
        Does not modify any attributes.
        """
        pass

    @abstractmethod
    def extract_diagnoses(self) -> str:
        """
        Extract the proposed diagnoses from the chat obtained from self.simulate().
        Does not modify any attributes.
        """
        pass

    @property
    def correct_diagnosis(self) -> str:
        return self.vignette.correct_diagnosis

    @timeit(logger, log_kwargs=False)
    def simulate(self) -> None:
        """
        Run the diagnosis simulation.
        Iteratively build self.chat_doctor and self.chat_patient by repeating the following:
            i) Infer new patient output based on existing chat (self.chat_patient)
            ii) Append it to both the chat_doctor and chat_patient
            iii) Infer new doctor output based on existing chat (self.chat_doctor)
            iv) Append it to both the chat_doctor and chat_patient
        Stop when the diagnosis is finished or too long.
        """
        out_doctor = self.chat_doctor.messages[-1]
        try:
            while True:
                out_patient = self.infer_patient()
                # Patient output has role ASSISTANT. The role needs to be changed to USER
                # before adding it to self.chat_doctor, to simulate what a user would say.
                prompt_doctor = gen_cmsg(out_doctor, role=Role.USER, body=out_patient.body)
                self.chat_patient.messages.append(out_patient)
                self.chat_doctor.messages.append(prompt_doctor)

                out_doctor = self.infer_doctor()
                # Doctor output has role ASSISTANT. The role needs to be changed to USER
                # before adding it to self.chat_patient, to simulate what a user would say.
                prompt_patient = gen_cmsg(out_patient, role=Role.USER, body=out_doctor.body)
                self.chat_doctor.messages.append(out_doctor)
                self.chat_patient.messages.append(prompt_patient)

                if self.diagnosis_finished or len(self.chat_patient) > self.max_len:
                    break
        except Exception:
            logger.exception(f"Error while simulating vignette {self.vignette}")

        # Set chat id from generated messages.
        if chat_id := self.chat_patient.messages[-1].chat_id:
            self.chat_patient.id = chat_id
            for msg in self.chat_patient.messages:
                msg.chat_id = chat_id
        if chat_id := self.chat_doctor.messages[-1].chat_id:
            self.chat_doctor.id = chat_id
            for msg in self.chat_doctor.messages:
                msg.chat_id = chat_id


class NaiveSimulator(Simulator):
    def infer_doctor(self) -> CMessage:
        if len(self.chat_doctor.messages) >= self.max_len - 4:
            last = self.chat_doctor.messages[-1]
            new = gen_cmsg(
                last,
                role=Role.SYSTEM,
                body="Immediately finish the conversation by listing the most likely diagnoses.",
            )
            self.chat_doctor.messages.append(new)
        return self.doctor_client.converse(self.chat_doctor.messages)

    def infer_patient(self) -> CMessage:
        return self.patient_client.converse(self.chat_patient.messages)

    @property
    def diagnosis_finished(self) -> bool:
        chat = self.chat_doctor
        return len(chat) > 3 and "DIAGNOSIS READY" in chat.messages[-1].body

    def extract_diagnoses(self) -> str:
        if self.diagnosis_finished:
            o = self.chat_doctor.messages[-1].body.split("DIAGNOSIS READY")[1]
            return o[: 1 + o.find("]")]
        else:
            return ""


class LocalSimulator(NaiveSimulator):
    def infer_doctor(self) -> CMessage:
        m, _ = marshal(self.chat_doctor.messages, rename_roles=True)
        # In the local server, the INSSS breaks the body into prompt and instruction.
        if len(self.chat_doctor.messages) < 15:
            m += (
                "INSSS"
                + f"""Below is the transcript of your current conversation with the patient.
                Ask the patient a question to find out more about their condition so that you can better diagnose the patient. Never repeat questions.
            """
            )
        else:
            m += (
                "INSSS"
                + """Below is the transcript of a differential diagnosis.  Based on the transcript, write 5 most likely diagnoses. Write 5 chosen diagnoses in this format:
            DIAGNOSIS READY: [diagnosis1, diagnosis2, diagnosis3, diagnosis4, diagnosis5]
            """
            )
        cmsg = CMessage(user_id=1, role=Role.USER, body=m)
        o = self.doctor_client.inquire(cmsg)
        o.body.replace("Response:\n", "")
        return o
