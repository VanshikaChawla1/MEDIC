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

🧑‍⚕️ Clinician Dashboard (React.js)
│
▼
🚀 FastAPI Backend (Python)
│
▼
🤖 Multi-Agent RL Engine (PyTorch)
│
▼
🪄 SHAP Explainability Layer
│
▼
🗄️ SQLite Database (Patients & Resources)


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
