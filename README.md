# Factory-Production-Line-Simulation
SimuLine is a discrete-event simulation of a multi-stage factory production line developed for the System Modeling and Simulation (CCS3003/CS305) course. The simulation models the flow of products through a series of machines with individual processing times and random failure/repair behavior.


# SimuLine: Factory Production Line Simulation

## 📌 Project Overview
This project simulates a factory assembly line to analyze performance metrics such as throughput, machine utilization, failure downtime, and bottlenecks. It was developed for the **System Modeling and Simulation (CCS3003/CS305)** course at the Arab Academy, under the supervision of **Prof. Dr. Khaled Mahar** and **TA Sara Mohamed**.

---

## 🎯 Objectives
- Simulate a production line with multiple machines.
- Model product flow, failures, and repairs using discrete-event simulation.
- Track key performance indicators like:
  - Product throughput
  - Machine downtimes
  - Waiting times
  - Bottlenecks

---

## 🛠️ Simulation Components

- **Entities**: Products moving through the production line.
- **Resources**: Machines with processing capacity and failure/repair behavior.
- **Failure Modeling**: Machines fail randomly and are repaired after a stochastic duration.
- **Event Types**:
  - Product Arrival
  - Machine Start/Finish Processing
  - Machine Failure
  - Machine Repair Completion

---

## 📊 Distributions Used
- **Interarrival Times**: Exponential Distribution
- **Processing Times**: Triangular Distribution
- **Failure/Repair Times**: Weibull and Exponential Distributions

---

## 📈 Data Collected
- Downtime per machine
- Number of products processed (throughput)
- Queue lengths and waiting times
- Machine utilization
- Identification of bottlenecks


---

## 📝 Report Checklist

- ✅ Cover page (Project title, team members, course, project type)
- ✅ Simulation table with at least 20 events
- ✅ Snapshot of code with explanations
- ✅ Discussion of performance results and bottlenecks

---

## 👥 Team Members
- Marwan Elsayed - 221003166
- Hussein Galal - 221001810

---

## 📅 Course Info
- **Course**: System Modeling and Simulation (CCS3003/CS305)
  
---

## ⭐ Bonus Features
- 📊 Data visualizations of queue length and machine utilization
- 💡 Creative modeling improvements (e.g., predictive maintenance logic)

---

## 📎 License
MIT License – free to use and adapt with credit.

