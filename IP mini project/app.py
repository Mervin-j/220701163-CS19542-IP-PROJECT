from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, session, flash
import cv2
from ultralytics import YOLO
import numpy as np
import mysql.connector

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# Load the YOLO model and set up video capture
model = YOLO('models/best.pt')
cap = cv2.VideoCapture('sample_video.mp4')

if not cap.isOpened():
    raise Exception("Could not open video device")

# Define constants (same as in your original code)
heavy_traffic_threshold = 10
vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)
x1, x2 = 325, 635
lane_threshold = 609

def create_connection():
    connection = mysql.connector.connect(
        host='localhost',
        user='host',  # Your MySQL username
        password='root',  # Your MySQL password
        database='traffic_analysis_db'  # Your database name
    )
    return connection

def process_frame():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create a copy of the original frame to modify
        detection_frame = frame.copy()
        detection_frame[:x1, :] = 0  # Black out from top to x1
        detection_frame[x2:, :] = 0  # Black out from x2 to the bottom of the frame
        
        # Perform inference on the modified frame
        results = model.predict(detection_frame, imgsz=640, conf=0.4)
        processed_frame = results[0].plot(line_width=1)
        
        # Restore the original top and bottom parts of the frame
        processed_frame[:x1, :] = frame[:x1, :].copy()
        processed_frame[x2:, :] = frame[x2:, :].copy()

        # Convert frame to JPEG for video streaming
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def get_traffic_data():
    # Capture a frame and process it to determine traffic data
    ret, frame = cap.read()
    if not ret:
        return {'error': 'No frame available'}
    
    # Same detection logic as in the original code
    detection_frame = frame.copy()
    detection_frame[:x1, :] = 0  # Black out from top to x1
    detection_frame[x2:, :] = 0  # Black out from x2 to the bottom of the frame

    # Perform inference on the modified frame
    results = model.predict(detection_frame, imgsz=640, conf=0.4)

    # Retrieve the bounding boxes from the results
    bounding_boxes = results[0].boxes
    vehicles_in_left_lane = 0
    vehicles_in_right_lane = 0

    for box in bounding_boxes.xyxy:
        if box[0] < lane_threshold:
            vehicles_in_left_lane += 1
        else:
            vehicles_in_right_lane += 1
    
    # Determine traffic intensity for both lanes
    traffic_intensity_left = "Heavy" if vehicles_in_left_lane > heavy_traffic_threshold else "Smooth"
    traffic_intensity_right = "Heavy" if vehicles_in_right_lane > heavy_traffic_threshold else "Smooth"

    return {
        'left_lane': {'vehicles': vehicles_in_left_lane, 'intensity': traffic_intensity_left},
        'right_lane': {'vehicles': vehicles_in_right_lane, 'intensity': traffic_intensity_right}
    }

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Validate credentials against the database
        conn = create_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()  # Fetch the user from the database

        print(f"User fetched from database: {user}")  # Debug: Check user fetched

        if user:
            print(f"Password from database: {user['password']}")  # Debug: Show stored password
            if user['password'] == password:  # Check if the password matches
                session['username'] = username  # Store the username in the session
                return redirect(url_for('index'))  # Redirect to the index page after successful login
            else:
                flash('Invalid username or password', 'error')  # Flash an error message if login fails
                return render_template('login.html', error='Invalid username or password')  # Render login page with error
        else:
            flash('Invalid username or password', 'error')
            return render_template('login.html', error='Invalid username or password')  # Render login page with error

    return render_template('login.html')  # Render the login page on GET requests

@app.route('/')
def index():
    if 'username' in session:
        return render_template('index.html')  # Render the index page if logged in
    else:
        return redirect(url_for('login'))  # Redirect to login page if not logged in

@app.route('/video_feed')
def video_feed():
    return Response(process_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/traffic_data')
def traffic_data():
    data = get_traffic_data()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
