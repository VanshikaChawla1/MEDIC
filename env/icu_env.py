# env/icu_env.py
from dataclasses import dataclass
from typing import List
import random

@dataclass
class Patient:
    id: int
    arrival_time: int
    severity: float
    treated: bool = False

class ICUEnv:
    def __init__(self):
        self.time = 0
        self.patients: List[Patient] = []
        self.treated_patients: List[Patient] = []

    def add_patient(self, severity: float):
        patient_id = len(self.patients) + 1
        patient = Patient(id=patient_id, arrival_time=self.time, severity=severity)
        self.patients.append(patient)

    def step(self):
        self.time += 1

    def get_pending_patients(self):
        return [p for p in self.patients if not p.treated]

    def treat_patient(self, patient_id: int):
        for p in self.patients:
            if p.id == patient_id and not p.treated:
                p.treated = True
                self.treated_patients.append(p)
                return p
        return None

    def reset(self):
        self.time = 0
        self.patients = []
        self.treated_patients = []

    def get_action_space(self):
        """Return number of possible actions (patients to choose)"""
        return len(self.get_pending_patients())

    def step_rl(self, action: int):
        """
        Treat the selected patient (action = patient index in pending list)
        Returns:
            next_state: severity list of waiting patients
            reward: severity of treated patient
            done: bool
        """
        pending = self.get_pending_patients()
        if not pending:
            return [], 0.0, True  # No patients left

        # Clamp action to valid index
        action = min(action, len(pending) - 1)

        patient = pending[action]
        self.treat_patient(patient.id)

        reward = patient.severity
        next_state = [p.severity for p in self.get_pending_patients()]
        done = len(next_state) == 0

        return next_state, reward, done

    def select_random_action(self):
        pending = self.get_pending_patients()
        if not pending:
            return 0
        return random.randint(0, len(pending) - 1)
class ICUMultiEnv(ICUEnv):
    def __init__(self, num_agents=2):
        super().__init__()
        self.num_agents = num_agents
        self.agents_done = [False]*num_agents

    def step_multi(self, actions):
        """
        actions: list of actions from all agents
        Returns: next_state, reward, done
        """
        total_reward = 0
        for action in actions:
            if len(self.get_pending_patients()) == 0:
                continue
            # Clamp action
            action = min(action, len(self.get_pending_patients())-1)
            _, reward, _ = self.step_rl(action)
            total_reward += reward
        done = len(self.get_pending_patients()) == 0
        next_state = [p.severity for p in self.get_pending_patients()]
        return next_state, total_reward, done
