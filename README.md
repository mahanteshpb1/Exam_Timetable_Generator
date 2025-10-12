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

Home Page
<img width="2747" height="1616" alt="image" src="https://github.com/user-attachments/assets/a0f2d543-e3cf-491c-be25-1676cde73406" />

Upload the input file
<img width="1829" height="1617" alt="image" src="https://github.com/user-attachments/assets/e404ab04-17b8-4b58-84dd-ecd790948993" />

Output
<img width="1743" height="1600" alt="image" src="https://github.com/user-attachments/assets/48870cad-a9e6-4811-97fc-9888bb2a01cf" />



