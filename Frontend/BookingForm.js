// frontend/medic-frontend/src/components/BookingForm.js
import React, { useState } from "react";
import axios from "axios";

function BookingForm({ setView }) {
  const [form, setForm] = useState({
    name: "",
    age: "",
    blood_pressure: "",
    sugar_level: "",
    cholesterol: "",
    symptoms: "",
  });
  const [result, setResult] = useState(null);
  const [errorInfo, setErrorInfo] = useState(null);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult(null);
    setErrorInfo(null);
    try {
      const res = await axios.post("http://127.0.0.1:8000/api/book", form);
      setResult(res.data);
    } catch (err) {
      // API returns 400 with missing resources info
      const data = err.response?.data;
      if (data && data.missing) {
        setErrorInfo({
          message: data.message || "Insufficient resources",
          missing: data.missing,
          resources: data.current_resources,
        });
      } else {
        alert("Error while booking patient. Check server logs.");
        console.error(err);
      }
    }
  };

  return (
    <div className="form-container">
      <button className="back-btn" onClick={() => setView("home")}>← Back</button>
      <h2>🧾 Patient Booking Form</h2>
      <form onSubmit={handleSubmit}>
        <input name="name" placeholder="Patient Name" onChange={handleChange} required />
        <input name="age" type="number" placeholder="Age" onChange={handleChange} required />
        <input name="blood_pressure" type="number" placeholder="Blood Pressure" onChange={handleChange} required />
        <input name="sugar_level" type="number" placeholder="Sugar Level" onChange={handleChange} required />
        <input name="cholesterol" type="number" placeholder="Cholesterol" onChange={handleChange} required />
        <textarea name="symptoms" placeholder="Symptoms" onChange={handleChange}></textarea>
        <button type="submit">Book Patient</button>
      </form>

      {result && (
        <div className="result-box">
          <h3>{result.message}</h3>
          <p><strong>Patient ID:</strong> {result.patient_id}</p>
          <p><strong>Predicted Score:</strong> {result.predicted_score}</p>
          <p><strong>Triage:</strong> {result.triage}</p>
          <p><strong>Confidence (est.):</strong> {result.confidence}</p>
          <h4>Remaining resources</h4>
          <pre>{JSON.stringify(result.resources, null, 2)}</pre>
        </div>
      )}

      {errorInfo && (
        <div className="result-box" style={{ background: "#ffecec" }}>
          <h3 style={{ color: "#c00" }}>{errorInfo.message}</h3>
          <p>Missing resources:</p>
          <pre>{JSON.stringify(errorInfo.missing, null, 2)}</pre>
          <p>Current resources:</p>
          <pre>{JSON.stringify(errorInfo.resources, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default BookingForm;
