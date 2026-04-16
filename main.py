from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import face_recognition
import cv2
import numpy as np
import pandas as pd
import os
import shutil
from datetime import datetime
from pathlib import Path
import uuid

app = FastAPI(title="Class Attendance System")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_CSV = "attendance.csv"

# Initialize
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
if not os.path.exists(ATTENDANCE_CSV):
    pd.DataFrame(columns=['name', 'roll_no', 'time']).to_csv(ATTENDANCE_CSV, index=False)

known_encodings = []
known_students = []  # (name, roll_no)

def load_known_faces():
    """Load all known student faces"""
    global known_encodings, known_students
    known_encodings.clear()
    known_students.clear()
    
    for student_dir in os.listdir(KNOWN_FACES_DIR):
        student_path = Path(KNOWN_FACES_DIR) / student_dir
        if student_path.is_dir():
            for img_file in student_path.glob("*.jpg"):
                try:
                    image = face_recognition.load_image_file(img_file)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        known_encodings.append(encodings[0])
                        known_students.append(student_dir)
                except:
                    pass
    print(f"Loaded {len(known_students)} known faces")

load_known_faces()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/enroll", response_class=HTMLResponse)
async def enroll_page(request: Request):
    return templates.TemplateResponse("enroll.html", {"request": request})

@app.post("/enroll")
async def enroll_student(
    name: str = Form(...),
    roll_no: str = Form(...),
    photo1: UploadFile = File(...),
    photo2: UploadFile = File(...),
    photo3: UploadFile = File(...)
):
    # Create student directory
    student_id = f"{name}_{roll_no}"
    student_dir = Path(KNOWN_FACES_DIR) / student_id
    student_dir.mkdir(exist_ok=True)
    
    # Save photos
    photos = [photo1, photo2, photo3]
    for i, photo in enumerate(photos):
        file_path = student_dir / f"face_{i+1}.jpg"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(photo.file, f)
    
    # Reload known faces
    load_known_faces()
    
    return {"message": f"✅ Enrolled {name} ({roll_no}) successfully!", "redirect": "/enroll"}

@app.get("/detect", response_class=HTMLResponse)
async def detect_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect")
async def detect_attendance(classroom_photo: UploadFile = File(...)):
    # Save classroom photo
    photo_path = f"classroom_{uuid.uuid4().hex}.jpg"
    with open(photo_path, "wb") as f:
        shutil.copyfileobj(classroom_photo.file, f)
    
    # Process photo
    image = face_recognition.load_image_file(photo_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    attendance = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        if matches and face_distances.min() < 0.6:
            best_match = np.argmin(face_distances)
            student = known_students[best_match]
            if student not in attendance:
                attendance.append(student)
                # Log to CSV
                df = pd.DataFrame({
                    'name': [student],
                    'roll_no': [student.split('_')[1]],
                    'time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                })
                df.to_csv(ATTENDANCE_CSV, mode='a', header=False, index=False)
    
    os.remove(photo_path)
    
    return {"attendance": attendance, "total_faces": len(face_encodings), "redirect": "/"}

@app.get("/attendance")
async def get_attendance():
    if os.path.exists(ATTENDANCE_CSV):
        df = pd.read_csv(ATTENDANCE_CSV)
        return df.tail(10).to_dict('records')
    return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)