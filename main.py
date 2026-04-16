from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import HTTPException
import mediapipe as mp
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

# MediaPipe setup
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1, 
    min_detection_confidence=0.5
)

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_CSV = "attendance.csv"

# Initialize
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
if not os.path.exists(ATTENDANCE_CSV):
    pd.DataFrame(columns=['name', 'roll_no', 'time', 'confidence']).to_csv(ATTENDANCE_CSV, index=False)

known_faces = {}  # {student_id: photo_paths}

def load_known_faces():
    """Load known student faces"""
    global known_faces
    known_faces.clear()
    if os.path.exists(KNOWN_FACES_DIR):
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
    latest_attendance = []
    if os.path.exists(ATTENDANCE_CSV):
        try:
            df = pd.read_csv(ATTENDANCE_CSV)
            latest_attendance = df.tail(5).to_dict('records')
        except:
            pass
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "latest_attendance": latest_attendance,
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
    # Sanitize student ID
    student_id = f"{name.replace(' ', '_').replace('-', '_')}_{roll_no}"
    student_dir = Path(KNOWN_FACES_DIR) / student_id
    student_dir.mkdir(exist_ok=True)
    
    # Save photos
    photos = [photo1, photo2, photo3]
    for i, photo in enumerate(photos):
        contents = await photo.read()
        file_path = student_dir / f"face_{i+1}.jpg"
        with open(file_path, "wb") as f:
            f.write(contents)
    
    load_known_faces()
    # Redirect back to enroll page with success
    raise HTTPException(status_code=303, headers={"Location": "/enroll"})
    
@app.get("/detect", response_class=HTMLResponse)
async def detect_page(request: Request):
    return templates.TemplateResponse("detect.html", {"request": request})

@app.post("/detect")
async def detect_attendance(classroom_photo: UploadFile = File(...)):
    # Save classroom photo temporarily
    photo_filename = f"classroom_{uuid.uuid4().hex}.jpg"
    photo_path = Path(photo_filename)
    contents = await classroom_photo.read()
    with open(photo_path, "wb") as f:
        f.write(contents)
    
    try:
        # Load and process image
        image = cv2.imread(str(photo_path))
        height, width, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # MediaPipe detection
        results = face_detection.process(rgb_image)
        
        attendance = []
        face_count = 0
        
        if results.detections:
            for detection in results.detections:
                face_count += 1
                
                # Bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Crop face
                face_crop = image[max(0, y-20):min(height, y+h+20), 
                                max(0, x-20):min(width, x+w+20)]
                
                if face_crop.size > 0:
                    face_crop_resized = cv2.resize(face_crop, (112, 112))
                    face_crop_path = f"temp_face_{face_count}.jpg"
                    cv2.imwrite(face_crop_path, face_crop_resized)
                    
                    # Match against known faces
                    matched_student = None
                    best_confidence = 0.0
                    
                    for student_id, photos in known_faces.items():
                        for known_photo_path in photos[:2]:
                            try:
                                known_img = cv2.imread(str(known_photo_path))
                                if known_img is not None:
                                    known_img = cv2.resize(known_img, (112, 112))
                                    
                                    # Template matching
                                    result = cv2.matchTemplate(known_img, face_crop_resized, cv2.TM_CCOEFF_NORMED)
                                    confidence = np.max(result)
                                    
                                    if confidence > 0.6 and confidence > best_confidence:
                                        best_confidence = confidence
                                        matched_student = student_id
                            except:
                                continue
                    
                    # Log match
                    if matched_student:
                        name_parts = matched_student.split('_')
                        student_name = '_'.join(name_parts[:-1]).replace('_', ' ')
                        roll_no = name_parts[-1]
                        
                        attendance.append({
                            "name": student_name,
                            "roll_no": roll_no,
                            "confidence": f"{best_confidence:.2f}"
                        })
                        
                        # Save to CSV
                        df_new = pd.DataFrame({
                            'name': [matched_student],
                            'roll_no': [roll_no],
                            'time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                            'confidence': [best_confidence]
                        })
                        df_new.to_csv(ATTENDANCE_CSV, mode='a', header=False, index=False)
                    
                    # Cleanup
                    if os.path.exists(face_crop_path):
                        os.remove(face_crop_path)
        
        # Redirect to home with results
        # For demo, return success redirect
        raise HTTPException(status_code=303, headers={"Location": "/"})
        
    finally:
        # Always cleanup
        if photo_path.exists():
            photo_path.unlink()

@app.get("/download")
async def download_csv():
    if os.path.exists(ATTENDANCE_CSV):
        return FileResponse(
            ATTENDANCE_CSV,
            media_type='text/csv',
            filename='class_attendance.csv'
        )
    raise HTTPException(status_code=404, detail="No attendance records found")

@app.get("/files")
async def list_files():
    return {
        "known_students": list(known_faces.keys()) if known_faces else [],
        "attendance_exists": os.path.exists(ATTENDANCE_CSV),
        "known_faces_count": len(known_faces)
    }

@app.get("/debug")
async def debug():
    import sys
    return {
        "python_version": sys.version,
        "mediapipe_version": mp.__version__,
        "known_students": len(known_faces),
        "directories": os.listdir(".")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)