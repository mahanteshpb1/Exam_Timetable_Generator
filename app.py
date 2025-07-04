from flask import Flask, render_template, request, redirect, url_for, session
from scheduler import InputData, TimeTable, SchedulerMain, Exam, Room, TimeSlot, ExamSlot, Chromosome, Student
import os
from typing import Dict, List # Import Dict and List for type hinting
import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24) # Used for session management

# DB Manager is no longer initialized

@app.route('/')
def index():
    # Direct link to the form page
    return redirect(url_for('form_page'))

@app.route('/form', methods=['GET', 'POST'])
def form_page():
    if request.method == 'POST':
        try:
            # Handle Excel file upload
            excel_file = request.files.get('excelFile')
            if not excel_file:
                raise ValueError("No Excel file uploaded.")
            
            # Save the uploaded file temporarily
            temp_excel_path = "uploaded_exam_data.xlsx"
            excel_file.save(temp_excel_path)

            # Retrieve other form data
            num_rooms = int(request.form.get('numRooms'))
            room_capacities = list(map(int, request.form.getlist('roomCapacities[]')))
            start_date_str = request.form.get('startDate')
            end_date_str = request.form.get('endDate')

            # Get custom room numbers from the form
            room_numbers = request.form.getlist('roomNumbers[]')
            if len(room_numbers) != num_rooms:
                raise ValueError("Number of room numbers provided does not match the number of rooms.")

            # Validate dates
            if not start_date_str or not end_date_str:
                raise ValueError("Start date and End date are required.")

            input_data_instance = InputData(
                excel_file_path=temp_excel_path,
                num_rooms=num_rooms,
                room_capacities=room_capacities,
                start_date_str=start_date_str,
                end_date_str=end_date_str,
                room_numbers=room_numbers
            )
            
            timetable_instance = TimeTable(input_data=input_data_instance)
            
            scheduler = SchedulerMain(input_data=input_data_instance, timetable=timetable_instance)
            
            if scheduler.final_chromosome:
                formatted_schedule = format_exam_schedule_for_display(scheduler.final_chromosome, input_data_instance)
                return render_template('view.html', timetable_data=formatted_schedule)
            else:
                error_message = "Could not generate an optimal exam schedule within the given generations."
                return render_template('form.html', error=error_message)
        except Exception as e:
            error_message = f"Error during exam schedule generation: {e}"
            return render_template('form.html', error=error_message)
    
    return render_template('form.html')

@app.route('/how-to-use')
def how_to_use():
    return render_template('how-to-use.html')

def format_exam_schedule_for_display(chromosome: Chromosome, input_data: InputData) -> Dict:
    """
    Formats the chromosome (exam schedule) data into a dictionary suitable for rendering in HTML.
    The structure will be organized by date, then hour, then room, listing exams and students in each slot.
    """
    scheduled_exams_by_timeslot: Dict[TimeSlot, Dict[Room, List[Exam]]] = {}
    exam_students_map: Dict[str, List[Student]] = {}

    # First, create a map of which students are taking each exam
    for student in input_data.students:
        for exam_code in student.enrolled_exams:
            if exam_code not in exam_students_map:
                exam_students_map[exam_code] = []
            exam_students_map[exam_code].append(student)

    for gene in chromosome.genes:
        exam_slot = chromosome.all_exam_slots[gene.assigned_exam_slot_id]
        if exam_slot.time_slot not in scheduled_exams_by_timeslot:
            scheduled_exams_by_timeslot[exam_slot.time_slot] = {}
        if exam_slot.room not in scheduled_exams_by_timeslot[exam_slot.time_slot]:
            scheduled_exams_by_timeslot[exam_slot.time_slot][exam_slot.room] = []
        scheduled_exams_by_timeslot[exam_slot.time_slot][exam_slot.room].append(exam_slot.exam)
    
    formatted_data = {}
    # Sort time slots for ordered output
    sorted_timeslots = sorted(scheduled_exams_by_timeslot.keys(), key=lambda ts: (ts.date, ts.hour))

    for ts in sorted_timeslots:
        date_str = ts.date.isoformat()
        hour_str = f"{ts.hour:02d}:00"

        if date_str not in formatted_data:
            formatted_data[date_str] = {}
        if hour_str not in formatted_data[date_str]:
            formatted_data[date_str][hour_str] = []
        
        sorted_rooms = sorted(scheduled_exams_by_timeslot[ts].keys(), key=lambda r: r.id)
        for room in sorted_rooms:
            exams_in_room = []
            for exam in scheduled_exams_by_timeslot[ts][room]:
                # Get students taking this exam
                students = exam_students_map.get(exam.code, [])
                student_info = [f"{s.name} ({s.roll_number})" for s in students]
                exams_in_room.append({
                    "code": exam.code,
                    "students": student_info
                })
            
            formatted_data[date_str][hour_str].append({
                "room_id": room.id,
                "capacity": room.capacity,
                "exams": exams_in_room
            })
        
        # Ensure all rooms are present for each time slot, even if no exams are scheduled
        for room_obj in input_data.rooms:
            if room_obj not in sorted_rooms:
                formatted_data[date_str][hour_str].append({
                    "room_id": room_obj.id,
                    "capacity": room_obj.capacity,
                    "exams": []
                })
        # Re-sort rooms within each hour after adding missing rooms
        formatted_data[date_str][hour_str].sort(key=lambda x: x['room_id'])
    
    return formatted_data

if __name__ == '__main__':
    # No input.txt creation needed anymore
    app.run(debug=True) 