<h1 align="center">🧠 MEDIC: Multi-Agent Explainable Decision-Making for Intelligent Care 🩺</h1>

<p align="center">
  <em>Bridging Intelligence, Transparency, and Care through Reinforcement Learning & Explainable AI</em><br>
  <strong>Multi-Agent Reinforcement Learning + SHAP Explainability + Real-Time Clinical Insights</strong>
</p>

---

## 🌟 Project Vision

Healthcare demands not only intelligent decisions — but also *transparent* ones.  
**MEDIC** (Multi-Agent Explainable Decision-making for Intelligent Care) is a next-generation AI system designed to **assist clinicians** in **real-time hospital resource allocation** using **Multi-Agent Reinforcement Learning (MARL)** combined with **Explainable AI (SHAP)**.

It learns to optimize ICU beds, doctor availability, and oxygen distribution dynamically — while explaining *why* each decision is made.

---

## 🩻 Why MEDIC?

> “A decision that can’t be explained isn’t a decision you can trust.”

Traditional AI systems act as black boxes. MEDIC changes this by:
- 🤝 Blending **Human + Machine Intelligence**
- 🧩 Using **Multi-Agent Reinforcement Learning** to coordinate limited hospital resources
- 🔍 Integrating **SHAP** for clear, interpretable decision explanations
- 🌐 Offering a **real-time web dashboard** for doctors and administrators

---

## 🧭 System Architecture

+-------------------------------------------------------------+
|                   🧠 MEDIC System Overview                  |
+-------------------------------------------------------------+
|                                                             |
|  👩‍⚕️ Clinician (User Interface - React.js)                |
|        │                                                    |
|        ▼                                                    |
|  🌐 Frontend Layer: Real-time Dashboard                     |
|        - Patient Booking Interface                          |
|        - Resource Monitoring Graphs                         |
|        - SHAP Explanation Visualization                     |
|                                                             |
|        │                                                    |
|        ▼                                                    |
|  ⚙️ Backend API Layer (FastAPI + Uvicorn)                   |
|        - Handles Requests and WebSocket/SSE Connections     |
|        - Performs Data Preprocessing and Validation          |
|                                                             |
|        │                                                    |
|        ▼                                                    |
|  🤖 Decision Engine (Multi-Agent RL System)                 |
|        - ICU Bed Agent                                      |
|        - Oxygen Resource Agent                              |
|        - Doctor Allocation Agent                            |
|        - Reward Function for Optimal Policy                 |
|                                                             |
|        │                                                    |
|        ▼                                                    |
|  🧩 Explainability Layer (SHAP Integration)                  |
|        - Computes Shapley Values for Each Decision           |
|        - Generates Patient-Level Explanation Graphs          |
|                                                             |
|        │                                                    |
|        ▼                                                    |
|  🗄️ Database Layer (SQLite)                                 |
|        - Stores Patient Data                                |
|        - Maintains Resource Availability                    |
|        - Logs Agent Decisions and SHAP Outputs              |
|                                                             |
+-------------------------------------------------------------+



> Agents collaborate to allocate ICU beds, doctors, and oxygen resources dynamically — while the SHAP engine explains each decision in real-time.

---

## ⚙️ Key Features

| Feature | Description |
|----------|-------------|
| 🧩 **Multi-Agent RL** | Independent agents for ICU, doctors, and oxygen collaborate for optimal resource allocation. |
| 🩻 **SHAP Explainability** | Every decision is explained via Shapley values for transparency. |
| 🌐 **Real-Time Updates** | Event-driven synchronization via Server-Sent Events (SSE). |
| 💻 **Interactive Dashboard** | Clean, React-based clinician interface with live SHAP visualization. |
| 🧠 **Smart Triage System** | Prioritizes patients dynamically based on risk and resource availability. |
| 🧾 **Admin Control** | Delete or manage patient records and auto-update resources in real time. |

---
⚙️ Tech Stack
Layer	Technology	Description
Frontend	React.js, Recharts	Real-time dashboard visualization
Backend	FastAPI, Python	Core MARL logic and REST APIs
Database	SQLite	Lightweight relational database
Explainability	SHAP	Model interpretability and patient-level transparency
Deployment	Uvicorn	Fast API server runtime

---

🖼️ Screenshots
🏥 Home Interface

<img width="940" height="302" alt="image" src="https://github.com/user-attachments/assets/6209b55e-93cc-434d-9f7c-62178e6aef6c" />

🩺 Clinician Dashboard

Displays:

Current resource levels (ICU, oxygen, doctors):

<img width="940" height="494" alt="image" src="https://github.com/user-attachments/assets/6ee47be3-229b-4172-9586-b3b9a1abdf9f" />

Real-time patient bookings:

<img width="940" height="447" alt="image" src="https://github.com/user-attachments/assets/c72c6f3d-a797-403d-ba2f-a4cd0818a51c" />

SHAP explainability chart showing patient priorities:

<img width="940" height="399" alt="image" src="https://github.com/user-attachments/assets/107c1a64-d90c-4707-88dc-d782c31ec523" />

🧮 Booking Form

<img width="940" height="524" alt="image" src="https://github.com/user-attachments/assets/7ccc0737-ae9e-4d8c-b950-0e45936fddba" />

---

📊 Results & Discussion 
- Multi-agent RL model effectively coordinated between agents to optimize limited hospital resources.
- Explainability integration via SHAP ensured each decision’s transparency, improving trust in AI-based recommendations.
- Frontend dashboard provided real-time, interpretable insights, enabling clinicians to make informed decisions in critical care.

---


