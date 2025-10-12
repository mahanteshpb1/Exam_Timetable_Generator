# 🧮 Exam Timetable Generator

A **Flask-based web application** that automatically generates an **optimized exam timetable** using a **Genetic Algorithm (GA)**.  
It schedules exams efficiently based on student enrollments, room capacities, and available dates — minimizing exam clashes and ensuring balanced utilization of rooms.

---

## 🚀 Features

- ✅ Automatic timetable generation using a Genetic Algorithm  
- ✅ Ensures clash-free scheduling for students  
- ✅ Customizable room numbers and capacities  
- ✅ Upload student–exam data via Excel file  
- ✅ Simple Flask-based web interface  
- ✅ Configurable exam date range  
- ✅ Displays results in a structured, readable timetable view  

---

## 🧰 Tech Stack

| Component      | Technology                         |
|----------------|------------------------------------|
| Frontend       | HTML, CSS, Jinja2 Templates        |
| Backend        | Python (Flask)                     |
| Algorithm      | Genetic Algorithm                  |
| Data Input     | Excel (.xlsx)                      |
| Libraries      | pandas, openpyxl, typing, datetime |





⚙️ Installation

1️⃣ Clone the Repository
```bash
git clone https://github.com/mahanteshpb1/Exam_Timetable_Generator.git
cd exam-timetable-generator

2️⃣ Create a Virtual Environment

python -m venv venv
# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate


3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the App
python app.py



