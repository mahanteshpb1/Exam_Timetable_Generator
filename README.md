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


## 📂 Folder Structure

exam-timetable-generator/
├── scheduler/
│ ├── init.py
│ ├── scheduler.py # Core scheduling & GA logic
│ ├── models.py # Data classes (Exam, Room, Student, etc.)
│ └── utils.py # Helper functions
├── templates/
│ ├── index.html
│ ├── form.html
│ ├── view.html
│ └── how-to-use.html
├── static/ # (optional) for CSS/JS
├── app.py # Flask entry point
├── requirements.txt
└── README.md
