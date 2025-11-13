import React, { useState } from "react";
import Home from "./components/Home";
import BookingForm from "./components/BookingForm";
import ClinicianDashboard from "./components/ClinicianDashboard";
import "./App.css";

function App() {
  const [view, setView] = useState("home");

  return (
    <div className="app-container">
      {view === "home" && <Home setView={setView} />}
      {view === "booking" && <BookingForm setView={setView} />}
      {view === "dashboard" && <ClinicianDashboard setView={setView} />}
    </div>
  );
}

export default App;
