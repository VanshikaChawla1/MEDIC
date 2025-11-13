import React from "react";

function Home({ setView }) {
  return (
    <div className="home-container">
      <h1>🏥 MEDIC System</h1>
      <p>Choose your role to proceed:</p>
      <div className="button-group">
        <button onClick={() => setView("booking")}>🧾 Patient Booking</button>
        <button onClick={() => setView("dashboard")}>🩺 Clinician Dashboard</button>
      </div>
    </div>
  );
}

export default Home;
