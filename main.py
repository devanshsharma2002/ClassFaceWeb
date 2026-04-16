from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import os
import shutil
from datetime import datetime
from pathlib import Path
from PIL import Image
import io
import uuid

app = FastAPI(title="Class Attendance System - MediaPipe Edition")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# MediaPipe setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    model_selection=1, 
    min_detection_confidence=0.5
)

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_CSV = "attendance.csv"

# Initialize directories
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
if not os.path.exists(ATTENDANCE_CSV):
    pd.DataFrame(columns=['name', 'roll_no', 'time', 'confidence']).to_csv(ATTENDANCE_CSV, index=False)

known_faces = {}  # {student_id: [face_paths]}

def load_known_faces():
    """Load known student faces"""
    global known_faces
    known_faces.clear()
    for student_dir in os.listdir(KNOWN_FACES_DIR):
        student_path = Path(KNOWN_FACES_DIR) / student_dir
        if student_path.is_dir():
            photos = list(student_path.glob("*.jpg"))
            if photos:
                known_faces[student_dir] = photos
    print(f"Loaded {len(known_faces)} known students")

load_known_faces()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    latest = pd.read_csv(ATTENDANCE_CSV).tail(5).to_dict('records') if os.path.exists(ATTENDANCE_CSV) else []
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "latest_attendance": latest,
        "known_count": len(known_faces)
    })

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
    student_id = f"{name.replace(' ', '_')}_{roll_no}"
    student_dir = Path(KNOWN_FACES_DIR) / student_id
    student_dir.mkdir(exist_ok=True)
    
    # Save 3 enrollment photos
    photos = [photo1, photo2, photo3]
    for i, photo in enumerate(photos):
        file_path = student_dir / f"face_{i+1}.jpg"
        contents = await photo.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    
    load_known_faces()
    return {"message": f"✅ Enrolled {name} ({roll_no}) with 3 photos!", "student_id": student_id}

@app.get("/detect", response_class=HTMLResponse)
async def detect_page(request: Request):
    return templates.TemplateResponse("detect.html", {"request": request})

@app.post("/detect")
async def detect_attendance(classroom_photo: UploadFile = File(...)):
    # Save uploaded classroom photo
    photo_path = f"classroom_{uuid.uuid4().hex}.jpg"
    contents = await classroom_photo.read()
    with open(photo_path, "wb") as f:
        f.write(contents)
    
    # MediaPipe face detection
    image = cv2.imread(photo_path)
    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = face_detection.process(rgb_image)
    
    attendance = []
    face_count = 0
    
    if results.detections:
        for detection in results.detections:
            face_count += 1
            
            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            
            # Crop face
            face_crop = image[y:y+h, x:x+w]
            face_crop_path = f"temp_face_{face_count}.jpg"
            cv2.imwrite(face_crop_path, face_crop)
            
            # Simple demo matching (compare with known faces)
            matched_student = None
            confidence = 0.0
            
            for student_id, photos in known_faces.items():
                for known_photo_path in photos[:2]:  # Check first 2 photos
                    try:
                        known_img = cv2.imread(str(known_photo_path))
                        known_img = cv2.resize(known_img, (100, 100))
                        face_img = cv2.resize(face_crop, (100, 100))
                        
                        # Simple template matching
                        result = cv2.matchTemplate(known_img, face_img, cv2.TM_CCOEFF_NORMED)
                        match_conf = np.max(result)
                        
                        if match_conf > 0.6 and match_conf > confidence:
                            confidence = match_conf
                            matched_student = student_id
                    except:
                        continue
            
            if matched_student:
                roll_no = matched_student.split('_')[1] if '_' in matched_student else "N/A"
                attendance.append({
                    "name": matched_student.split('_')[0].replace('_', ' '),
                    "roll_no": roll_no,
                    "confidence": f"{confidence:.2f}"
                })
                
                # Log to CSV
                df = pd.DataFrame({
                    'name': [matched_student],
                    'roll_no': [roll_no],
                    'time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'confidence': [confidence]
                })
                df.to_csv(ATTENDANCE_CSV, mode='a', header=False, index=False)
            
            os.remove(face_crop_path)
    
    os.remove(photo_path)
    
    return {
        "attendance": attendance,
        "total_faces_detected": face_count,
        "total_known_students": len(known_faces)
    }

@app.get("/download")
async def download_csv():
    if os.path.exists(ATTENDANCE_CSV):
        return FileResponse(
            ATTENDANCE_CSV,
            media_type='text/csv',
            filename='attendance.csv'
        )
    raise HTTPException(status_code=404, detail="No attendance data")

@app.get("/debug")
async def debug():
    import sys
    return {
        "python_version": sys.version,
        "known_students": list(known_faces.keys()),
        "mediapipe_version": mp.__version__
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)