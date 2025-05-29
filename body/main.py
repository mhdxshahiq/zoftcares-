import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import sqlite3
import numpy as np
import os
import time
import datetime
import random
import logging
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import threading
import cv2  # OpenCV for webcam
from PIL import Image, ImageTk  # Pillow for image conversion
from queue import Queue

# --- Configuration ---
LOG_LEVEL = logging.INFO
FEATURE_VECTOR_SIZE = 64
FEATURE_DATA_TYPE = np.float64

DB_FILE_APP = "app_employee_system_v2.db"
CLEAR_DB_ON_START_APP = False

USE_COSINE_SIMILARITY_APP = True
GAIT_THRESHOLD_COS_APP = 0.80
BSHAPE_THRESHOLD_COS_APP = 0.80
COMBINED_THRESHOLD_COS_APP = 0.82
GAIT_THRESHOLD_EUC_APP = 0.7
BSHAPE_THRESHOLD_EUC_APP = 0.7
COMBINED_THRESHOLD_EUC_APP = 0.7
GAIT_SCORE_WEIGHT_APP = 0.5
BSHAPE_SCORE_WEIGHT_APP = 0.5

TIME_WINDOW_SECONDS_APP = 7  # Not directly used in webcam GUI but part of engine
UNKNOWN_PERSON_THRESHOLD_SIM_APP = 0.55
UNKNOWN_PERSON_THRESHOLD_DIST_APP = 1.2

WEBCAM_DISPLAY_WIDTH = 640
WEBCAM_DISPLAY_HEIGHT = 480
SIMULATED_DETECTION_INTERVAL = 3 # seconds

# --- Tkinter Logging Handler ---
class TkinterLogHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.queue = Queue()
        self.text_widget.after(100, self._process_queue)

    def emit(self, record):
        msg = self.format(record)
        self.queue.put(msg)

    def _process_queue(self):
        while not self.queue.empty():
            try:
                msg = self.queue.get_nowait()
                self.text_widget.configure(state='normal')
                self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.configure(state='disabled')
                self.text_widget.see(tk.END)
            except Queue.Empty: # Should not happen due to while not self.queue.empty() but good practice
                pass
            except Exception as e:
                print(f"Error in _process_queue: {e}") # Fallback
        self.text_widget.after(100, self._process_queue)

# --- EmployeeDatabase Class ---
class EmployeeDatabase:
    def __init__(self, db_name=DB_FILE_APP):
        self.db_name = db_name
        self._conn = None
        self._cursor = None
        self._connect()
        self._create_table()
        # logging.info(f"EmployeeDatabase initialized with {self.db_name}") # Logging done globally

    def _connect(self):
        self._conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self._cursor = self._conn.cursor()

    def _create_table(self):
        try:
            self._cursor.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    employee_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    stored_gait_features BLOB,
                    stored_bodyshape_features BLOB
                )
            """)
            self._conn.commit()
        except Exception as e:
            logging.error(f"Error creating table: {e}", exc_info=True)
            raise

    def add_employee(self, employee_data):
        try:
            gait_features_blob = employee_data["stored_gait_features"].astype(FEATURE_DATA_TYPE).tobytes()
            bodyshape_features_blob = employee_data["stored_bodyshape_features"].astype(FEATURE_DATA_TYPE).tobytes()
            self._cursor.execute("""
                INSERT INTO employees (employee_id, name, stored_gait_features, stored_bodyshape_features)
                VALUES (?, ?, ?, ?)
            """, (employee_data["employee_id"], employee_data["name"], gait_features_blob, bodyshape_features_blob))
            self._conn.commit()
            # logging.info(f"Employee {employee_data['employee_id']} added.") # Logging done in App
            return True
        except sqlite3.IntegrityError:
            # logging.warning(f"Employee ID {employee_data['employee_id']} already exists.")
            return False
        except Exception as e:
            logging.error(f"Error adding employee {employee_data['employee_id']}: {e}", exc_info=True)
            return False

    def get_employee(self, employee_id):
        try:
            self._cursor.execute("SELECT * FROM employees WHERE employee_id = ?", (employee_id,))
            row = self._cursor.fetchone()
            if row:
                return {
                    "employee_id": row[0], "name": row[1],
                    "stored_gait_features": np.frombuffer(row[2], dtype=FEATURE_DATA_TYPE),
                    "stored_bodyshape_features": np.frombuffer(row[3], dtype=FEATURE_DATA_TYPE)
                }
            return None
        except Exception as e:
            logging.error(f"Error retrieving employee {employee_id}: {e}", exc_info=True)
            return None

    def get_all_employees(self):
        try:
            self._cursor.execute("SELECT * FROM employees")
            rows = self._cursor.fetchall()
            employees = []
            for row in rows:
                employees.append({
                    "employee_id": row[0], "name": row[1],
                    "stored_gait_features": np.frombuffer(row[2], dtype=FEATURE_DATA_TYPE),
                    "stored_bodyshape_features": np.frombuffer(row[3], dtype=FEATURE_DATA_TYPE)
                })
            return employees
        except Exception as e:
            logging.error(f"Error retrieving all employees: {e}", exc_info=True)
            return []
    
    def get_employee_count(self):
        try:
            self._cursor.execute("SELECT COUNT(*) FROM employees")
            count = self._cursor.fetchone()[0]
            return count
        except Exception as e:
            logging.error(f"Error getting employee count: {e}", exc_info=True)
            return 0
            
    def clear_database_table(self):
        try:
            self._cursor.execute("DELETE FROM employees")
            self._conn.commit()
            logging.info(f"All records cleared from employees table in {self.db_name}.")
        except Exception as e:
            logging.error(f"Error clearing employees table: {e}", exc_info=True)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
            self._cursor = None
            # logging.debug(f"Database connection to {self.db_name} closed.")

    def __del__(self):
        self.close()

# --- BiometricMatcher Class ---
class BiometricMatcher: # Unchanged
    def __init__(self, gait_threshold=0.85, bodyshape_threshold=0.85, combined_threshold=0.85,
                 use_cosine_similarity=True, gait_weight=0.5, bodyshape_weight=0.5):
        self.gait_threshold = gait_threshold
        self.bodyshape_threshold = bodyshape_threshold
        self.combined_threshold = combined_threshold
        self.use_cosine_similarity = use_cosine_similarity
        self.gait_weight = gait_weight
        self.bodyshape_weight = bodyshape_weight

    def _calculate_score(self, features1, features2):
        if features1 is None or features2 is None or features1.size == 0 or features2.size == 0:
            return 0.0 if self.use_cosine_similarity else float('inf')
        if features1.ndim == 1: features1 = features1.reshape(1, -1)
        if features2.ndim == 1: features2 = features2.reshape(1, -1)
        if features1.shape[1] != features2.shape[1]:
            return 0.0 if self.use_cosine_similarity else float('inf')
        if self.use_cosine_similarity:
            return cosine_similarity(features1, features2)[0][0]
        else:
            return euclidean_distances(features1, features2)[0][0]

    def match_gait(self, extracted_gait, stored_gait):
        score = self._calculate_score(extracted_gait, stored_gait)
        match = (score >= self.gait_threshold) if self.use_cosine_similarity else (score <= self.gait_threshold)
        return score, match

    def match_bodyshape(self, extracted_bodyshape, stored_bodyshape):
        score = self._calculate_score(extracted_bodyshape, stored_bodyshape)
        match = (score >= self.bodyshape_threshold) if self.use_cosine_similarity else (score <= self.bodyshape_threshold)
        return score, match

    def combined_match_score(self, gait_score, bodyshape_score):
        if not (isinstance(gait_score, (int, float)) and isinstance(bodyshape_score, (int,float))):
            return 0.0 if self.use_cosine_similarity else float('inf')
        if np.isinf(gait_score) or np.isinf(bodyshape_score):
             return 0.0 if self.use_cosine_similarity else float('inf')
        return (gait_score * self.gait_weight) + (bodyshape_score * self.bodyshape_weight)

    def evaluate_combined_match(self, combined_score):
        return (combined_score >= self.combined_threshold) if self.use_cosine_similarity else (combined_score <= self.combined_threshold)

# --- IdentificationEngine Class ---
class IdentificationEngine:
    def __init__(self, employee_db, biometric_matcher, app_controller, # app_controller for general status
                 time_window_seconds=TIME_WINDOW_SECONDS_APP,
                 unknown_person_threshold_similarity=UNKNOWN_PERSON_THRESHOLD_SIM_APP,
                 unknown_person_threshold_distance=UNKNOWN_PERSON_THRESHOLD_DIST_APP):
        self.employee_db = employee_db
        self.biometric_matcher = biometric_matcher
        self.app_controller = app_controller 
        # self.cctv_event_buffer = [] # Less relevant for pure webcam mode without badge scans
        self.time_window_seconds = time_window_seconds
        self.unknown_person_threshold_similarity = unknown_person_threshold_similarity
        self.unknown_person_threshold_distance = unknown_person_threshold_distance

    def process_simulated_detection(self, cctv_event, source="Webcam"):
        # This method will return details for on-frame drawing
        logging.info(f"Processing {source} Detection: Person {cctv_event['person_id_in_video']}")
        extracted_gait = cctv_event["extracted_gait_features"]
        extracted_bodyshape = cctv_event["extracted_bodyshape_features"]
        all_employees = self.employee_db.get_all_employees()

        identification_result = {"name": "Unknown", "id": None, "status_type": "UNKNOWN", "color": "red"}

        if not all_employees:
            logging.warning(f"{source} Detection: No employees in database.")
            self.app_controller.update_detection_status("No employees in DB", "red") # General status
            return identification_result

        best_match_info = {
            "employee_id": None, "name": "Unknown",
            "gait_score": 0.0 if self.biometric_matcher.use_cosine_similarity else float('inf'),
            "bodyshape_score": 0.0 if self.biometric_matcher.use_cosine_similarity else float('inf'),
            "combined_score": 0.0 if self.biometric_matcher.use_cosine_similarity else float('inf'),
            "gait_match": False, "bodyshape_match": False
        }

        for emp in all_employees:
            stored_gait, stored_bodyshape = emp["stored_gait_features"], emp["stored_bodyshape_features"]
            gait_score, gait_match = self.biometric_matcher.match_gait(extracted_gait, stored_gait)
            bodyshape_score, bodyshape_match = self.biometric_matcher.match_bodyshape(extracted_bodyshape, stored_bodyshape)
            current_combined_score = self.biometric_matcher.combined_match_score(gait_score, bodyshape_score)
            
            is_better = (current_combined_score > best_match_info["combined_score"]) if self.biometric_matcher.use_cosine_similarity else \
                        (current_combined_score < best_match_info["combined_score"])
            if is_better:
                best_match_info = {"employee_id": emp["employee_id"], "name": emp["name"],
                                   "gait_score": gait_score, "bodyshape_score": bodyshape_score,
                                   "combined_score": current_combined_score,
                                   "gait_match": gait_match, "bodyshape_match": bodyshape_match}

        status_msg_for_gui_label = "Unknown Person"
        status_color_for_gui_label = "red"

        if best_match_info["employee_id"]:
            identification_result["id"] = best_match_info["employee_id"]
            if best_match_info["gait_match"] and best_match_info["bodyshape_match"]:
                identification_result["name"] = best_match_info["name"]
                identification_result["status_type"] = "IDENTIFIED"
                identification_result["color"] = "green"
                status_msg_for_gui_label = f"IDENTIFIED: {best_match_info['name']}"
                status_color_for_gui_label = "green"
                logging.info(f"{status_msg_for_gui_label} via {source}. Scores: G={best_match_info['gait_score']:.2f}, BS={best_match_info['bodyshape_score']:.2f}")
            elif (self.biometric_matcher.use_cosine_similarity and best_match_info["combined_score"] >= self.unknown_person_threshold_similarity) or \
                 (not self.biometric_matcher.use_cosine_similarity and best_match_info["combined_score"] <= self.unknown_person_threshold_distance):
                identification_result["name"] = best_match_info["name"] + "?"
                identification_result["status_type"] = "POTENTIAL"
                identification_result["color"] = "orange"
                status_msg_for_gui_label = f"POTENTIAL: {best_match_info['name']}"
                status_color_for_gui_label = "orange"
                logging.warning(f"{status_msg_for_gui_label} via {source}. Scores: G={best_match_info['gait_score']:.2f}, BS={best_match_info['bodyshape_score']:.2f}")
            else: # Low score
                status_msg_for_gui_label = f"Unknown (Best guess: {best_match_info['name']} - Low Score)"
                logging.info(f"{status_msg_for_gui_label} via {source}. Person {cctv_event['person_id_in_video']}.")
        else: # No significant match at all
            logging.info(f"Unknown Person via {source} (Person ID: {cctv_event['person_id_in_video']}). No match.")
        
        self.app_controller.update_detection_status(status_msg_for_gui_label, status_color_for_gui_label)
        return identification_result


# --- Helper Functions ---
def generate_random_features(size=FEATURE_VECTOR_SIZE):
    return np.random.rand(size).astype(FEATURE_DATA_TYPE)

def get_timestamp_str(dt_obj=None):
    if dt_obj is None: dt_obj = datetime.datetime.now()
    return dt_obj.strftime("%Y-%m-%d %H:%M:%S")

# --- Main Application Class ---
class BiometricApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Biometric Employee Identification System v2")
        self.root.geometry("1000x750")

        self.employee_db = EmployeeDatabase()
        matcher_params = {"use_cosine_similarity": USE_COSINE_SIMILARITY_APP,
                          "gait_weight": GAIT_SCORE_WEIGHT_APP, "bodyshape_weight": BSHAPE_SCORE_WEIGHT_APP}
        if USE_COSINE_SIMILARITY_APP: # Configure thresholds based on metric
            matcher_params.update({"gait_threshold": GAIT_THRESHOLD_COS_APP, "bodyshape_threshold": BSHAPE_THRESHOLD_COS_APP, "combined_threshold": COMBINED_THRESHOLD_COS_APP})
        else:
            matcher_params.update({"gait_threshold": GAIT_THRESHOLD_EUC_APP, "bodyshape_threshold": BSHAPE_THRESHOLD_EUC_APP, "combined_threshold": COMBINED_THRESHOLD_EUC_APP})
        self.matcher = BiometricMatcher(**matcher_params)
        self.engine = IdentificationEngine(self.employee_db, self.matcher, self)

        self.is_realtime_detection_active = False
        self.webcam_thread = None
        self.video_capture = None
        self.person_id_counter = 0 
        self.last_simulated_person_info = None # To store info for drawing on frame

        self._setup_ui()
        self._setup_logging() # Call this after UI elements it uses are created
        logging.info("Application V2 initialized.")
        self.update_db_status()

        if CLEAR_DB_ON_START_APP:
            self.employee_db.clear_database_table()
            logging.info("Database cleared on start as per configuration.")
            self.update_db_status()

    def _setup_logging(self): # Logging setup
        self.log_text = scrolledtext.ScrolledText(self.logging_frame, state='disabled', height=10, wrap=tk.WORD, bg="#ECECEC", fg="black")
        self.log_text.pack(expand=True, fill='both', padx=5, pady=5)
        
        logger = logging.getLogger()
        logger.setLevel(LOG_LEVEL)
        for handler in logger.handlers[:]: logger.removeHandler(handler) # Clear existing
            
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M:%S')
        
        self.tk_log_handler = TkinterLogHandler(self.log_text)
        self.tk_log_handler.setFormatter(formatter)
        logger.addHandler(self.tk_log_handler)
        logging.info("Logging redirected to GUI.")

    def _setup_ui(self): # UI Setup
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(expand=True, fill='both')

        # Left Panel for Registration and Controls
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill='y', padx=(0,10), anchor='nw')
        
        reg_frame = ttk.LabelFrame(left_panel, text="Employee Registration", padding="10")
        reg_frame.pack(fill='x', pady=(0,10))
        ttk.Label(reg_frame, text="Employee ID:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.emp_id_entry = ttk.Entry(reg_frame, width=25)
        self.emp_id_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(reg_frame, text="Employee Name:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.emp_name_entry = ttk.Entry(reg_frame, width=25)
        self.emp_name_entry.grid(row=1, column=1, padx=5, pady=5)
        self.register_button = ttk.Button(reg_frame, text="Register (Simulated Biometrics)", command=self.register_employee)
        self.register_button.grid(row=2, column=0, columnspan=2, padx=5, pady=10)
        self.db_status_label = ttk.Label(reg_frame, text="DB Status: Initializing...")
        self.db_status_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        # Right Panel for Webcam and Logs
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.LEFT, expand=True, fill='both')

        webcam_controls_frame = ttk.LabelFrame(right_panel, text="Real-time Detection Control", padding="10")
        webcam_controls_frame.pack(fill='x', pady=(0,5))
        self.start_detection_button = ttk.Button(webcam_controls_frame, text="Start Detection", command=self.start_realtime_detection)
        self.start_detection_button.pack(side=tk.LEFT, padx=5)
        self.stop_detection_button = ttk.Button(webcam_controls_frame, text="Stop Detection", command=self.stop_realtime_detection, state=tk.DISABLED)
        self.stop_detection_button.pack(side=tk.LEFT, padx=5)
        self.detection_status_label = ttk.Label(webcam_controls_frame, text="Status: Idle", font=("Helvetica", 12, "bold"), anchor="w")
        self.detection_status_label.pack(side=tk.LEFT, padx=10, expand=True, fill='x')

        webcam_display_frame = ttk.LabelFrame(right_panel, text="Webcam Feed", padding="5")
        webcam_display_frame.pack(expand=True, fill='both', pady=5)
        self.webcam_label = ttk.Label(webcam_display_frame) # No placeholder text, size will dictate
        self.webcam_label.pack(expand=True, fill='both', anchor='center')
        self.webcam_label.configure(background='black')

        self.logging_frame = ttk.LabelFrame(right_panel, text="System Log", padding="5")
        self.logging_frame.pack(fill='x', pady=(5,0), ipady=5, side=tk.BOTTOM, expand=False, anchor='sw')
        # Log text widget added in _setup_logging

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_db_status(self):
        count = self.employee_db.get_employee_count()
        self.db_status_label.config(text=f"DB: {count} employees")

    def register_employee(self):
        emp_id = self.emp_id_entry.get().strip()
        emp_name = self.emp_name_entry.get().strip()
        if not emp_id or not emp_name:
            messagebox.showerror("Error", "Employee ID and Name cannot be empty.")
            return
        gait_features = generate_random_features()
        bodyshape_features = generate_random_features()
        logging.info(f"Simulated biometric capture for {emp_id}.")
        employee_data = {"employee_id": emp_id, "name": emp_name,
                         "stored_gait_features": gait_features, "stored_bodyshape_features": bodyshape_features}
        if self.employee_db.add_employee(employee_data):
            logging.info(f"Employee {emp_name} ({emp_id}) registered.")
            messagebox.showinfo("Success", f"Employee {emp_name} registered (simulated biometrics).")
            self.emp_id_entry.delete(0, tk.END); self.emp_name_entry.delete(0, tk.END)
            self.update_db_status()
        else:
            logging.error(f"Failed to register employee {emp_id}. It might already exist.")
            messagebox.showerror("Error", f"Failed to register {emp_id}. May already exist.")

    def start_realtime_detection(self):
        if self.is_realtime_detection_active:
            logging.warning("Detection already active."); return
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            messagebox.showerror("Webcam Error", "Could not open webcam."); self.video_capture = None; return
        
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_DISPLAY_WIDTH)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_DISPLAY_HEIGHT)

        self.is_realtime_detection_active = True
        self.start_detection_button.config(state=tk.DISABLED)
        self.stop_detection_button.config(state=tk.NORMAL)
        self.register_button.config(state=tk.DISABLED)
        logging.info("Real-time detection started (simulation).")
        self.update_detection_status("Detection Active...", "blue")
        self.webcam_thread = threading.Thread(target=self._realtime_detection_loop, daemon=True)
        self.webcam_thread.start()

    def _realtime_detection_loop(self):
        last_simulated_detection_time = time.time()
        current_frame_detections = [] # Store {"box": (x,y,w,h), "text": "Name", "color": (B,G,R)}

        while self.is_realtime_detection_active and self.video_capture:
            ret, frame = self.video_capture.read()
            if not ret: logging.error("Failed to grab frame."); time.sleep(0.1); continue
            
            frame_for_display = frame.copy() # Work on a copy for drawing

            # Simulate a new detection periodically
            current_time = time.time()
            if current_time - last_simulated_detection_time > SIMULATED_DETECTION_INTERVAL:
                last_simulated_detection_time = current_time
                current_frame_detections.clear() # Clear old detections for this frame cycle
                
                # Simulate one "person" detection
                logging.debug("Simulating new person detection & biometric extraction...")
                sim_gait_features = generate_random_features()
                sim_bodyshape_features = generate_random_features()
                self.person_id_counter += 1
                person_video_id = f"Webcam_P{self.person_id_counter:03d}"
                cctv_event = {"type": "webcam_detection", "timestamp": get_timestamp_str(),
                              "location": "Live Webcam Feed", "person_id_in_video": person_video_id,
                              "extracted_gait_features": sim_gait_features,
                              "extracted_bodyshape_features": sim_bodyshape_features}
                
                identification_info = self.engine.process_simulated_detection(cctv_event)
                
                # Generate random bounding box for this simulated person
                fw, fh = frame.shape[1], frame.shape[0]
                box_w, box_h = random.randint(fw//6, fw//3), random.randint(fh//4, fh//2)
                box_x, box_y = random.randint(0, fw - box_w), random.randint(0, fh - box_h)
                
                text_to_display = identification_info["name"]
                color_bgr = (0,0,255) # Red for unknown by default
                if identification_info["status_type"] == "IDENTIFIED": color_bgr = (0,255,0) # Green
                elif identification_info["status_type"] == "POTENTIAL": color_bgr = (0,165,255) # Orange
                
                current_frame_detections.append({
                    "box": (box_x, box_y, box_w, box_h),
                    "text": text_to_display,
                    "color": color_bgr
                })

            # Draw all current detections for this frame
            for det in current_frame_detections:
                x, y, w, h = det["box"]
                cv2.rectangle(frame_for_display, (x, y), (x + w, y + h), det["color"], 2)
                cv2.putText(frame_for_display, det["text"], (x, y - 10 if y > 20 else y + h + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, det["color"], 2)

            try: # Update webcam label in GUI
                cv2image = cv2.cvtColor(frame_for_display, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                
                # Maintain aspect ratio for display
                img_w, img_h = img.size
                label_w, label_h = WEBCAM_DISPLAY_WIDTH, WEBCAM_DISPLAY_HEIGHT # Target display dimensions
                
                scale = min(label_w / img_w, label_h / img_h)
                new_w, new_h = int(img_w * scale), int(img_h * scale)
                img_resized = img.resize((new_w, new_h), Image.LANCZOS)
                
                imgtk = ImageTk.PhotoImage(image=img_resized)
                self.root.after(0, self._update_webcam_label, imgtk)
            except Exception as e:
                logging.error(f"Error processing/displaying webcam frame: {e}", exc_info=True)
            time.sleep(1/30) # Aim for ~30 FPS display

        if self.video_capture: self.video_capture.release(); self.video_capture = None
        self.root.after(0, self._clear_webcam_label)
        logging.info("Webcam detection loop finished.")

    def _update_webcam_label(self, imgtk):
        if self.is_realtime_detection_active:
            self.webcam_label.imgtk = imgtk
            self.webcam_label.config(image=imgtk, text="", width=WEBCAM_DISPLAY_WIDTH, height=WEBCAM_DISPLAY_HEIGHT)
            self.webcam_label.configure(background=None) # Remove placeholder bg
        else: self._clear_webcam_label()
            
    def _clear_webcam_label(self):
        self.webcam_label.config(image='', text="Webcam feed stopped.")
        self.webcam_label.configure(background='black', width=WEBCAM_DISPLAY_WIDTH, height=WEBCAM_DISPLAY_HEIGHT)


    def stop_realtime_detection(self):
        if not self.is_realtime_detection_active: logging.warning("Detection not active."); return
        self.is_realtime_detection_active = False
        if self.webcam_thread and self.webcam_thread.is_alive(): self.webcam_thread.join(timeout=1.0)
        if self.video_capture: self.video_capture.release(); self.video_capture = None
        self.start_detection_button.config(state=tk.NORMAL)
        self.stop_detection_button.config(state=tk.DISABLED)
        self.register_button.config(state=tk.NORMAL)
        self._clear_webcam_label()
        logging.info("Real-time detection stopped.")
        self.update_detection_status("Detection Stopped", "black")
        self.last_simulated_person_info = None # Clear any lingering info

    def update_detection_status(self, message, color="black"):
        def _update():
            self.detection_status_label.config(text=f"Status: {message}", foreground=color)
        if hasattr(self.root, 'after'): self.root.after(0, _update)
        else: logging.info(f"GUI Status Update (root gone): {message}")

    def on_closing(self):
        logging.info("Application closing...")
        if self.is_realtime_detection_active: self.stop_realtime_detection()
        if self.employee_db: self.employee_db.close()
        # Wait for non-daemon threads if necessary (tk_log_handler queue poller uses root.after, should be fine)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = BiometricApp(root)
    try:
        root.mainloop()
    except Exception as e:
        logging.critical(f"Unhandled exception in Tkinter mainloop: {e}", exc_info=True)
    finally:
        logging.info("Application has been shut down.")