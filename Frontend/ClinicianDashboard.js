// frontend/medic-frontend/src/components/ClinicianDashboard.js
import React, { useEffect, useState } from "react";
import axios from "axios";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Cell,
} from "recharts";

function ClinicianDashboard({ setView }) {
  const [resources, setResources] = useState({});
  const [bookings, setBookings] = useState([]);
  const [xaiResult, setXaiResult] = useState(null);

  // --- Fetch data from backend ---
  const fetchResources = async () => {
    try {
      const res = await axios.get("http://127.0.0.1:8000/api/resources");
      setResources(res.data.resources || {});
    } catch (err) {
      console.error("Error fetching resources:", err);
    }
  };

  const fetchBookings = async () => {
    try {
      const res = await axios.get("http://127.0.0.1:8000/api/bookings?limit=20");
      setBookings(res.data.bookings || []);
    } catch (err) {
      console.error("Error fetching bookings:", err);
    }
  };

  const fetchShap = async () => {
    try {
      const res = await axios.get("http://127.0.0.1:8000/xai/explain_all");
      const shapValues =
        res.data?.shap_values?.map((v) => (isFinite(Number(v)) ? Number(v) : 0)) || [];
      setXaiResult({
        ...res.data,
        shap_values: shapValues,
      });
    } catch (err) {
      console.error("Error fetching SHAP data:", err);
    }
  };

  // --- Real-time SSE ---
  useEffect(() => {
    fetchResources();
    fetchBookings();
    fetchShap();

    const es = new EventSource("http://127.0.0.1:8000/events");
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "init") {
          setResources(data.resources || {});
        } else if (data.type === "booking_created") {
          const booking = {
            created_at: new Date((data.timestamp || Date.now() / 1000) * 1000).toLocaleString(),
            name: data.name,
            triage: data.triage,
            score: data.score,
            resources: data.resources,
            patient_id: data.patient_id,
          };
          setBookings((prev) => [booking, ...prev].slice(0, 20));
          setResources(data.resources || {});
          fetchShap();
        } else if (data.type === "patient_deleted") {
          setResources(data.resources || {});
          fetchBookings();
          fetchShap();
        }
      } catch (err) {
        console.warn("SSE parse error:", err);
      }
    };
    es.onerror = (err) => {
      console.warn("SSE connection error:", err);
      es.close();
    };
    return () => es.close();
  }, []);

  // --- Chart Data ---
  const resourceData = [
    { name: "ICU Beds", value: resources.icu_beds || 0 },
    { name: "Oxygen Cylinders", value: resources.oxygen_cylinders || 0 },
    { name: "Doctors", value: resources.doctors || 0 },
  ];

  const recommendedIndex = xaiResult?.recommended_patient ?? -1;

  const shapData =
    bookings.map((b, idx) => ({
      id: b.patient_id,
      name: b.name,
      triage: b.triage,
      score: b.score,
      shap: xaiResult?.shap_values?.[idx] || 0,
    })) || [];

  // --- UI ---
  return (
    <div style={styles.container}>
      <button style={styles.backBtn} onClick={() => setView("home")}>
        ← Back
      </button>
      <h2 style={styles.heading}>🩺 Clinician Dashboard (Realtime)</h2>

      {/* ---- Current Resources ---- */}
      <div style={styles.card}>
        <h3>Current Resources</h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={resourceData} margin={{ top: 20, right: 20, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="value" fill="#007bff" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ---- Recent Bookings ---- */}
      <div style={styles.card}>
        <h3>Recent Bookings</h3>
        <div style={styles.tableWrapper}>
          <table style={styles.table}>
            <thead>
              <tr>
                <th>Time</th>
                <th>Name</th>
                <th>Triage</th>
                <th>Score</th>
                <th>Resources</th>
              </tr>
            </thead>
            <tbody>
              {bookings.map((b, i) => (
                <tr key={i}>
                  <td>{b.created_at || new Date().toLocaleString()}</td>
                  <td>{b.name}</td>
                  <td>{b.triage}</td>
                  <td>{Number(b.score || 0).toFixed(3)}</td>
                  <td>
                    <pre style={styles.inlinePre}>
                      {JSON.stringify(b.resources || {}, null, 0)}
                    </pre>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ---- Admin Panel ---- */}
      <div style={styles.card}>
        <h3>Admin Panel</h3>
        <p style={{ marginBottom: 10 }}>Manage patient records (delete if needed)</p>
        <div style={styles.tableWrapper}>
          <table style={styles.table}>
            <thead>
              <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Triage</th>
                <th>Score</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {bookings.map((b, i) => (
                <tr key={i}>
                  <td>{b.patient_id || i + 1}</td>
                  <td>{b.name}</td>
                  <td>{b.triage}</td>
                  <td>{Number(b.score || 0).toFixed(3)}</td>
                  <td>
                    <button
                      style={styles.deleteBtn}
                      onClick={async () => {
                        if (window.confirm(`Delete patient ${b.name}?`)) {
                          try {
                            const res = await axios.delete(
                              `http://127.0.0.1:8000/api/patient/${b.patient_id}`
                            );
                            alert(res.data.message);
                            setResources(res.data.resources || {});
                            fetchBookings();
                            fetchShap();
                          } catch (err) {
                            alert("Error deleting patient: " + err.message);
                          }
                        }
                      }}
                    >
                      🗑 Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ---- RL Explanation (SHAP) ---- */}
      <div style={styles.card}>
        <h3>RL Explanation (SHAP)</h3>
        {xaiResult && bookings.length > 0 ? (
          <>
            <p style={{ marginTop: 10, fontSize: "15px" }}>
              Recommended Next Patient to Treat:{" "}
              <strong style={{ color: "#007bff" }}>
                Patient ID #{bookings[recommendedIndex]?.patient_id || "?"} (
                {bookings[recommendedIndex]?.name || "Unknown"})
              </strong>
            </p>

            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={shapData}
    margin={{ top: 20, right: 20, left: 10, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="id" tick={{ fontSize: 12 }} label={{ value: "Patient ID", position: "bottom" }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip
                  formatter={(value, name, props) => [
                    value.toFixed(4),
                    `SHAP Value | Name: ${props.payload.name}, Triage: ${props.payload.triage}, Score: ${props.payload.score.toFixed(3)}`,
                  ]}
                />
                <Legend verticalAlign="top" height={36}/>
                <Bar dataKey="shap" radius={[8, 8, 0, 0]}>
                  {shapData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={index === recommendedIndex ? "#28a745" : "#ff7f7f"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </>
        ) : (
          <p>Loading SHAP explanation...</p>
        )}
      </div>
    </div>
  );
}

// ---- Styles ----
const styles = {
  container: {
    maxWidth: 900,
    margin: "30px auto",
    padding: "20px",
    background: "#f9f9ff",
    borderRadius: "16px",
    boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
  },
  heading: { textAlign: "center", marginBottom: 20 },
  card: {
    background: "#fff",
    borderRadius: 12,
    padding: 16,
    marginBottom: 24,
    boxShadow: "0 2px 6px rgba(0,0,0,0.05)",
  },
  tableWrapper: { overflowX: "auto", maxHeight: 250, overflowY: "auto" },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    textAlign: "center",
    fontSize: "14px",
  },
  inlinePre: { background: "transparent", whiteSpace: "nowrap" },
  backBtn: {
    background: "transparent",
    border: "none",
    color: "#007bff",
    cursor: "pointer",
    fontSize: "16px",
    marginBottom: 10,
  },
  deleteBtn: {
    background: "#ff6961",
    color: "white",
    border: "none",
    padding: "5px 10px",
    borderRadius: "6px",
    cursor: "pointer",
  },
};

export default ClinicianDashboard;
