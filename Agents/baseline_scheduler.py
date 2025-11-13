from typing import List
from env.icu_env import Patient  # <-- use the same class as the env

def fifo_scheduler(patients: List[Patient]):
    return sorted(patients, key=lambda x: x.arrival_time)

def severity_scheduler(patients: List[Patient]):
    return sorted(patients, key=lambda x: x.severity, reverse=True)


# Example test
if __name__ == "__main__":
    patients = [
        Patient(id=1, arrival_time=1, severity=2.5),
        Patient(id=2, arrival_time=2, severity=3.2),
        Patient(id=3, arrival_time=3, severity=1.8),
    ]

    print("FIFO schedule:")
    for p in fifo_scheduler(patients):
        print(f"Patient {p.id} (arrival={p.arrival_time}, severity={p.severity})")

    print("\nSeverity-based schedule:")
    for p in severity_scheduler(patients):
        print(f"Patient {p.id} (arrival={p.arrival_time}, severity={p.severity})")

