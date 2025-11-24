import tensorflow as tf
import cv2, os, csv, pickle
import numpy as np
from datetime import datetime
from mtcnn import MTCNN

model = tf.keras.models.load_model("fine_tuned_model.h5")
with open("real_names.pkl", "rb") as f:
    class_names = pickle.load(f)

detector = MTCNN()
def predict_face(img_path):
    img = img_path
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    x, y, w, h = faces[0]['box']   # take first face
    face = img_rgb[y:y+h, x:x+w]
    face = cv2.resize(face, (128,128))
    face = np.expand_dims(face, axis=0)
    pred = model.predict(face) 
    idx = np.argmax(pred[0])
    confidence = pred[0][idx]
    label = class_names[idx]
    print("Raw prediction vector:", pred[0])
    print("Argmax index:", np.argmax(pred[0]))
    print(f"Predicted: {class_names[idx]} ({100*np.max(pred[0]):.2f}% confidence)")
    return label, confidence

# 4. Setup Excel workbook
def init_csv(path = "attendance.csv"):
    if not os.path.exists(path):
        with open(path, "w", newline=" ") as f:
            writer =csv.writer(f)
            writer.writerow(["Name", "Confidence", "Date", "Time", "Status", "Total_Break_Time", "Remarks"])
    return path

# check if already marked today
def already_marked_today(name, path="attendance.csv"):
    today = datetime.now().strftime("%Y-%m-%d")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            if row[0] == name and row[2] == today and row[4] == "Login":
                return True
    return False

def get_status(now):
    time_str = now.strftime("%H:%M")
    if "09:00" <= time_str <= "09:30":
        return "Login"
    elif time_str>= "18:00":
        return "Logout"
    else:
        return "Break"

def log_attendance(label, confidence, path="attendance.csv"):
    now = datetime.now()
    # First mark of day -- Login
    if not already_marked_today(label, path):
        status = "Login"
    else:
        status = get_status(now)

    remarks = ""
    if status == "Login" and now.strftime("%H:%M") > "09:30":
        remarks = "Late"
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label, f"{confidence:.2f}", now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), status, "" ,remarks])
    print(f"{status} marked for {label} at {now} ({remarks})")

def calculate_total_break(name, path="attendance.csv"):
    today = datetime.now().strftime("%Y-%m-%d")
    times = []
    with open(path, 'r', encoding= "utf-8") as f:
        reader  = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >=5 and row[0] == name and row[2] == today and row[4] == "Break":
                times.append(datetime.strptime(row[3], "%H:%M:%S"))
    total_break = 0
    for i in range(0, len(times)-1, 2):
        out_time = times[i]
        in_time = times[i+1]
        total_break += (in_time - out_time).seconds
    return total_break//60  #Converted to Minutes

def update_break_time(path="attendance.csv"):
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    names = set(row[0] for row in rows if row[2] == today)
    for name in names:
        minutes = calculate_total_break(name, path)
        for row in rows:
            if row[0] == name and row[2]== today:
                if len(row)<7:
                    row.extend(["", ""])               
                row[5] = str(minutes)
                if minutes >30:
                    row[6] = "Break time exceeded"
                elif row[6] == "":
                    row[6] = ""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print("Total break time will be updated at 6 PM")

# 5. Webcam inference + logging
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    path = init_csv("attendance.csv")
    break_updated = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        label, confidence = predict_face(frame)
        # Overlay prediction
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if label and confidence > 0.75:
            log_attendance(label, confidence, path)
        # else:
        #     print(f"Attendance for {label} already marked today")   
        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        now = datetime.now().strftime("%H:%M")
        if now >= "18:00" and not break_updated:
            update_break_time(path)
            break_updated = True


    cap.release()
    cv2.destroyAllWindows()