# scheduler.py

import copy  # For deep cloning
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Set

import pandas as pd

# --- New Data Classes for Exam Scheduling ---

@dataclass
class Student:
    roll_number: str
    name: str
    enrolled_exams: Set[str] = field(default_factory=set) # Set of exam codes the student is taking

@dataclass
class Exam:
    code: str
    # Potentially add duration, required room type, etc. later if needed

@dataclass
class Room:
    id: str
    capacity: int

    def __hash__(self):
        return hash((self.id, self.capacity))

    def __eq__(self, other):
        if not isinstance(other, Room):
            return False
        return self.id == other.id and self.capacity == other.capacity

@dataclass(order=True) # order=True allows direct comparison for sorting
class TimeSlot:
    date: date
    hour: int # Represents the start hour of the slot, e.g., 9 for 9:00-10:00

    def __hash__(self):
        return hash((self.date, self.hour))

    def __eq__(self, other):
        if not isinstance(other, TimeSlot):
            return False
        return self.date == other.date and self.hour == other.hour

@dataclass
class ExamSlot:
    exam: Exam
    room: Room
    time_slot: TimeSlot
    # Potentially add students assigned to this specific slot later

# --- Input Data Handling (Redesigned for Exam Scheduling) ---

class InputData:
    students: List[Student] = field(default_factory=list)
    exams: List[Exam] = field(default_factory=list) # List of unique exams
    rooms: List[Room] = field(default_factory=list)
    num_rooms: int = 0
    room_capacity: int = 0
    start_date: date = date.today()
    end_date: date = date.today()
    
    # New attributes for exam scheduling
    num_exams: int = 0
    total_possible_exam_slots: int = 0
    def __init__(self, excel_file_path: str, num_rooms: int, room_capacities: List[int], start_date_str: str, end_date_str: str, room_numbers=None):
        self.excel_file_path = excel_file_path
        self.num_rooms = num_rooms
        self.room_capacities = room_capacities
        self.start_date = date.fromisoformat(start_date_str)
        self.end_date = date.fromisoformat(end_date_str)
        self.room_numbers = room_numbers if room_numbers is not None else list(range(num_rooms))
    
        self._load_data_from_excel()
        self._generate_rooms()
        self.num_exams = len(self.exams)

    def _load_data_from_excel(self):
        try:
            df = pd.read_excel(self.excel_file_path)
            
            required_columns = ['Student Name', 'Roll Number', 'Exam Code']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Excel file must contain columns: {required_columns}")
            students_dict: Dict[str, Student] = {}
            exams_set: Set[str] = set() # To collect unique exam codes
            for index, row in df.iterrows():
                roll_number = str(row['Roll Number'])
                student_name = str(row['Student Name'])
                exam_code = str(row['Exam Code'])
                
                if roll_number not in students_dict:
                    students_dict[roll_number] = Student(roll_number=roll_number, name=student_name)
                
                students_dict[roll_number].enrolled_exams.add(exam_code)
                exams_set.add(exam_code)
            
            self.students = list(students_dict.values())
            self.exams = [Exam(code=exam_code) for exam_code in sorted(list(exams_set))] # Sort for consistency

        except FileNotFoundError:
            print(f"Error: Excel file '{self.excel_file_path}' not found.")
            raise
        except ValueError as ve:
            print(f"Error processing Excel file: {ve}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while reading Excel: {e}")
            raise

    def _generate_rooms(self):
        self.rooms = [Room(id=str(self.room_numbers[i]), capacity=int(self.room_capacities[i])) for i in range(self.num_rooms)]



# --- Global TimeTable Slots Generation (Redesigned for Exam Scheduling) ---

class TimeTable:
    # Define typical exam hours. Assuming 9-10, 11-12, 2-3, 4-5 as example slots
    EXAM_HOURS = [9, 11, 14, 16] 
    slots: List[ExamSlot] = field(default_factory=list)
    
    def __init__(self, input_data: InputData):
        self.input_data = input_data
        self._generate_exam_slots()
        self.input_data.total_possible_exam_slots = len(self.slots)

    def _generate_exam_slots(self):
        """
        Generates a global list of all possible ExamSlots (exam, room, time_slot combinations)
        based on the input dates and rooms.
        """
        self.slots = []
        current_date = self.input_data.start_date

        while current_date <= self.input_data.end_date:
            for hour in self.EXAM_HOURS:
                time_slot = TimeSlot(date=current_date, hour=hour)
                for room in self.input_data.rooms:
                    # For each exam, create an ExamSlot for every possible time_slot and room
                    # This creates the pool of potential slots that the GA will arrange
                    for exam in self.input_data.exams:
                        self.slots.append(ExamSlot(exam=exam, room=room, time_slot=time_slot))
            current_date += timedelta(days=1)

    def get_slots(self) -> List[ExamSlot]:
        return self.slots


# --- Genetic Algorithm Components for Exam Scheduling ---

@dataclass
class Gene:
    """
    Represents the assignment of a specific exam instance to a specific ExamSlot.
    A chromosome will contain one gene for each unique exam.
    """
    exam_id: int # Index of the exam in InputData.exams list
    assigned_exam_slot_id: int # Index into the global TimeTable.slots list

    def deep_clone(self):
        return copy.deepcopy(self)


@dataclass
class Chromosome:
    input_data: InputData
    all_exam_slots: List[ExamSlot] # All possible exam slots generated by TimeTable
    genes: List[Gene] = field(default_factory=list)
    fitness: float = 0.0
    
    def __post_init__(self):
        # Initialize genes: each exam is assigned a random exam slot initially
        for i, exam in enumerate(self.input_data.exams):
            # Randomly assign an exam to a possible exam_slot_id
            random_slot_id = random.randrange(len(self.all_exam_slots))
            self.genes.append(Gene(exam_id=i, assigned_exam_slot_id=random_slot_id))
        self.calculate_fitness()

    def deep_clone(self):
        return copy.deepcopy(self)

    def calculate_fitness(self):
        """
        Calculates the fitness of the chromosome (exam schedule).
        Hard Constraints:
        1. No student can take two exams at the same time slot.
        2. Room capacity must not be exceeded for any exam slot.
        """
        violations = 0

        # --- Constraint 1: No student overlapping exams ---
        # Map (time_slot, student_roll_number) to number of exams
        student_time_slot_exams: Dict[TimeSlot, Dict[str, int]] = {}
        for gene in self.genes:
            assigned_exam_slot = self.all_exam_slots[gene.assigned_exam_slot_id]
            exam_code = assigned_exam_slot.exam.code
            time_slot = assigned_exam_slot.time_slot

            # Find all students taking this exam
            for student in self.input_data.students:
                if exam_code in student.enrolled_exams:
                    if time_slot not in student_time_slot_exams:
                        student_time_slot_exams[time_slot] = {}
                    
                    if student.roll_number not in student_time_slot_exams[time_slot]:
                        student_time_slot_exams[time_slot][student.roll_number] = 0
                    
                    student_time_slot_exams[time_slot][student.roll_number] += 1

        for time_slot_dict in student_time_slot_exams.values():
            for student_roll, exam_count in time_slot_dict.items():
                if exam_count > 1:
                    violations += (exam_count - 1) * 10 # High penalty for overlapping exams

        # --- Constraint 2: Room capacity exceeded ---
        # Map (time_slot, room_id) to list of exams scheduled in it
        room_time_slot_exams: Dict[TimeSlot, Dict[int, List[Exam]]] = {}
        for gene in self.genes:
            assigned_exam_slot = self.all_exam_slots[gene.assigned_exam_slot_id]
            room = assigned_exam_slot.room
            time_slot = assigned_exam_slot.time_slot
            exam = assigned_exam_slot.exam

            if time_slot not in room_time_slot_exams:
                room_time_slot_exams[time_slot] = {}
            if room.id not in room_time_slot_exams[time_slot]:
                room_time_slot_exams[time_slot][room.id] = []
            
            room_time_slot_exams[time_slot][room.id].append(exam)

        for time_slot_dict in room_time_slot_exams.values():
            for room_id, exams_in_room in time_slot_dict.items():
                # Get the actual room object to check its capacity
                current_room = next((r for r in self.input_data.rooms if r.id == room_id), None)
                if current_room and len(exams_in_room) > current_room.capacity:
                    # This penalty logic needs to consider how many students are assigned to each exam
                    # For now, a simplified penalty based on number of exams over capacity
                    violations += (len(exams_in_room) - current_room.capacity) * 5 # Penalty for over capacity
        
        # --- Soft Constraint: Exams distribution (optional for now) ---
        # Could add penalties for: exams too close together for a student, unbalanced room usage

        # Calculate fitness based on violations. Higher fitness is better (fewer violations).
        max_possible_violations = 1000 # A configurable large number to normalize fitness
        if violations == 0:
            self.fitness = 1.0 # Perfect score if no violations
        else:
            self.fitness = 1.0 / (1.0 + violations) # Invert violations for fitness
        

    def print_schedule(self):
        """Prints the generated exam schedule in a readable format."""
        print("\n--- Generated Exam Schedule ---")
        scheduled_exams_by_timeslot: Dict[TimeSlot, Dict[Room, List[Exam]]] = {}

        for gene in self.genes:
            exam_slot = self.all_exam_slots[gene.assigned_exam_slot_id]
            if exam_slot.time_slot not in scheduled_exams_by_timeslot:
                scheduled_exams_by_timeslot[exam_slot.time_slot] = {}
            if exam_slot.room not in scheduled_exams_by_timeslot[exam_slot.time_slot]:
                scheduled_exams_by_timeslot[exam_slot.time_slot][exam_slot.room] = []
            scheduled_exams_by_timeslot[exam_slot.time_slot][exam_slot.room].append(exam_slot.exam)
        
        # Sort time slots for ordered output
        sorted_timeslots = sorted(scheduled_exams_by_timeslot.keys())

        for ts in sorted_timeslots:
            print(f"\nDate: {ts.date}, Hour: {ts.hour}:00")
            sorted_rooms = sorted(scheduled_exams_by_timeslot[ts].keys(), key=lambda r: r.id)
            for room in sorted_rooms:
                exams_in_room = ", ".join([e.code for e in scheduled_exams_by_timeslot[ts][room]])
                print(f"  Room {room.id} (Capacity: {room.capacity}): {exams_in_room}")

    def print_chromosome(self):
        """Prints the raw gene assignments for debugging."""
        print("Raw Chromosome (Exam ID -> Assigned Exam Slot ID):")
        for gene in self.genes:
            print(f"  Exam {self.input_data.exams[gene.exam_id].code} -> Slot {gene.assigned_exam_slot_id}")

    def __lt__(self, other):
        return self.fitness > other.fitness # Higher fitness is 'less than' for sorting


class SchedulerMain:
    first_list: List[Chromosome] = field(default_factory=list)
    new_list: List[Chromosome] = field(default_factory=list)
    first_list_fitness: float = 0.0
    new_list_fitness: float = 0.0
    population_size: int = 200 # Smaller population for initial testing
    max_generations: int = 50 # Fewer generations for initial testing
    
    final_chromosome: Optional[Chromosome] = None
    crossover_rate: float = 0.8 # Define here, or pass from InputData
    mutation_rate: float = 0.1 # Define here, or pass from InputData

    def __init__(self, input_data: InputData, timetable: TimeTable):
        self.input_data = input_data
        self.timetable = timetable
        
        print("SchedulerMain initialized for Exam Scheduling.")
        self.initialise_population()
        self.create_new_generations()

    def initialise_population(self):
        """Generates the first generation of chromosomes."""
        self.first_list = []
        self.first_list_fitness = 0.0

        for _ in range(self.population_size):
            chromosome = Chromosome(input_data=self.input_data, all_exam_slots=self.timetable.get_slots())
            # Fitness is calculated in __post_init__ of Chromosome, but recalculate if needed
            chromosome.calculate_fitness()
            self.first_list.append(chromosome)
            self.first_list_fitness += chromosome.fitness
        
        self.first_list.sort(reverse=True) # Sort in descending order of fitness
        print("----------Initial Generation-----------\n")
        self._print_generation(self.first_list)

    def create_new_generations(self):
        """Generates newer generations using crossover and mutation."""
        nogenerations = 0

        while nogenerations < self.max_generations:
            self.new_list = []
            self.new_list_fitness = 0.0
            
            # Elitism: first 10% chromosomes added as is
            num_elite = self.population_size // 10
            for i in range(num_elite):
                elite_chromosome = self.first_list[i].deep_clone()
                self.new_list.append(elite_chromosome)
                self.new_list_fitness += elite_chromosome.fitness

            # Add other members after performing crossover and mutation
            i = num_elite
            while i < self.population_size:
                father = self.select_parent_roulette()
                mother = self.select_parent_roulette()

                # Crossover
                if random.random() < self.crossover_rate:
                    son = self.crossover(father, mother)
                else:
                    son = father.deep_clone() # If no crossover, clone father
                
                # Mutation
                if random.random() < self.mutation_rate:
                    self.mutate(son)
                
                son.calculate_fitness() # Recalculate fitness after mutation

                if son.fitness == 1.0: # Ideal fitness (no violations)
                    print("Selected Chromosome is:-")
                    son.print_chromosome()
                    self.final_chromosome = son
                    break # Found optimal solution
                
                self.new_list.append(son)
                self.new_list_fitness += son.fitness
                i += 1

            if self.final_chromosome: # Optimal solution found
                print("****************************************************************************************")
                print(f"\n\nOptimal Exam Schedule generated in generation {nogenerations+1} with fitness 1.0.")
                self.final_chromosome.print_schedule()
                break
            
            # If optimal chromosome not found in this generation
            self.first_list = self.new_list
            self.first_list.sort(reverse=True) # Sort the new generation by fitness
            print(f"**************************     Generation {nogenerations+2}     ********************************************\n")
            self._print_generation(self.first_list)
            nogenerations += 1

        if not self.final_chromosome:
            print("\nCould not find an optimal timetable within the given generations. Displaying the best found.")
            if self.first_list:
                self.final_chromosome = self.first_list[0] # The best chromosome from the last generation
                self.final_chromosome.print_schedule()
            else:
                print("No chromosomes were generated.")

    def select_parent_roulette(self) -> Chromosome:
        """
        Selects a parent using Roulette Wheel Selection from the current population.
        """
        total_fitness = sum(c.fitness for c in self.first_list)
        if total_fitness == 0: # Avoid division by zero if all fitnesses are zero
            return random.choice(self.first_list).deep_clone() # Pick a random one if all fitnesses are zero

        pick = random.uniform(0, total_fitness)
        current = 0.0
        for chromosome in self.first_list:
            current += chromosome.fitness
            if current > pick:
                return chromosome.deep_clone()
        
        return self.first_list[-1].deep_clone() # Fallback, should not be reached


    def mutate(self, chromosome: Chromosome):
        """
        Mutates a gene in the chromosome by reassigning an exam to a different random exam slot.
        """
        gene_to_mutate_idx = random.randrange(len(chromosome.genes))
        
        # Assign a new random exam slot to this gene
        new_assigned_exam_slot_id = random.randrange(len(self.timetable.get_slots()))
        chromosome.genes[gene_to_mutate_idx].assigned_exam_slot_id = new_assigned_exam_slot_id


    def crossover(self, father: Chromosome, mother: Chromosome) -> Chromosome:
        """
        Performs a uniform crossover operation. Each gene is inherited from either father or mother randomly.
        """
        child_genes = []
        for i in range(len(father.genes)):
            if random.random() < 0.5: # 50% chance to inherit from father
                child_genes.append(father.genes[i].deep_clone())
            else:
                child_genes.append(mother.genes[i].deep_clone())
        
        child = Chromosome(input_data=self.input_data, all_exam_slots=self.timetable.get_slots(), genes=child_genes)
        child.calculate_fitness()
        return child

    def _print_generation(self, chromosome_list: List[Chromosome]):
        """
        Prints important details of a generation.
        """
        print("Fetching details from this generation...\n")

        # To print only initial 4 chromosomes of sorted list
        num_to_print = min(4, len(chromosome_list))
        for i in range(num_to_print):
            print(f"Chromosome no.{i}: Fitness = {chromosome_list[i].fitness}")
            # chromosome_list[i].print_chromosome() # Can be uncommented for detailed debugging
            print("")
        
        if chromosome_list:
            print(f"Most fit chromosome from this generation has fitness = {chromosome_list[0].fitness}\n")
        else:
            print("No chromosomes in this generation.\n")