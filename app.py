from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import os
import cv2
import numpy as np
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from face_recognition_module import FaceRecognitionSystem

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'cybernights_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db = SQLAlchemy(app)

# Database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'admin', 'teacher'

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    roll_number = db.Column(db.String(20), unique=True, nullable=False)
    face_encoding = db.Column(db.Text, nullable=True)  # JSON string of face encoding
    
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    lecture_id = db.Column(db.Integer, nullable=False)
    date = db.Column(db.Date, nullable=False)
    status = db.Column(db.String(20), nullable=False)  # 'present', 'absent', 'leave'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    student = db.relationship('Student', backref=db.backref('attendances', lazy=True))

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_face_encoding(image_path):
    face_system = FaceRecognitionSystem()
    face_encoding = face_system.encode_face_image(image_path)
    return face_encoding

# Routes
@app.route('/')
def index():
    return app.send_static_file('index.html')
    
@app.route('/face-attendance')
def face_attendance():
    return app.send_static_file('face-attendance.html')
    
@app.route('/student-registration')
def student_registration():
    return app.send_static_file('student-registration.html')
    
@app.route('/attendance-report')
def attendance_report():
    return app.send_static_file('attendance-report.html')
    
@app.route('/face-recognition-test')
def face_recognition_test():
    return app.send_static_file('face-recognition-test.html')

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    user = User.query.filter_by(username=username).first()
    
    if user and user.password == password:  # In production, use proper password hashing
        session['user_id'] = user.id
        return jsonify({
            'success': True,
            'user': {
                'id': user.id,
                'name': user.name,
                'role': user.role
            }
        })
    
    return jsonify({'success': False, 'message': 'Invalid credentials'})

@app.route('/api/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True})

# Lecture model
class Lecture(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    course_code = db.Column(db.String(20), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow().date)
    
@app.route('/api/lectures', methods=['GET'])
def get_lectures():
    lectures = Lecture.query.all()
    return jsonify({
        'success': True,
        'lectures': [
            {
                'id': lecture.id,
                'name': lecture.name,
                'course_code': lecture.course_code,
                'date': lecture.date.strftime('%Y-%m-%d') if lecture.date else None
            } for lecture in lectures
        ]
    })

@app.route('/api/students/register', methods=['POST'])
def register_student():
    """Register a new student with face recognition"""
    if 'name' not in request.form or 'roll_number' not in request.form:
        return jsonify({'success': False, 'message': 'Name and roll number are required'})
    
    if 'face_image' not in request.files:
        return jsonify({'success': False, 'message': 'Face image is required'})
    
    name = request.form['name']
    roll_number = request.form['roll_number']
    email = request.form.get('email', '')
    phone = request.form.get('phone', '')
    
    # Check if student with roll number already exists
    existing_student = Student.query.filter_by(roll_number=roll_number).first()
    if existing_student:
        # If student exists but doesn't have face encoding, update it
        if not existing_student.face_encoding:
            face_image = request.files['face_image']
            if face_image and allowed_file(face_image.filename):
                filename = f"{roll_number}_face.jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                face_image.save(filepath)
                
                # Get face encoding
                face_encoding = get_face_encoding(filepath)
                
                if not face_encoding:
                    return jsonify({'success': False, 'message': 'No face detected in the image. Please try again.'})
                
                # Update student
                existing_student.face_encoding = json.dumps(face_encoding)
                db.session.commit()
                
                return jsonify({
                    'success': True,
                    'message': 'Student face data updated successfully',
                    'student': {
                        'id': existing_student.id,
                        'name': existing_student.name,
                        'roll_number': existing_student.roll_number
                    }
                })
            return jsonify({'success': False, 'message': 'Invalid file format'})
        else:
            return jsonify({'success': False, 'message': 'Student with this roll number already exists'})
    
    # Save face image
    face_image = request.files['face_image']
    if face_image and allowed_file(face_image.filename):
        filename = f"{roll_number}_face.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        face_image.save(filepath)
        
        # Get face encoding
        face_encoding = get_face_encoding(filepath)
        
        if not face_encoding:
            return jsonify({'success': False, 'message': 'No face detected in the image. Please try again.'})
        
        # Create new student
        student = Student(
            name=name,
            roll_number=roll_number,
            face_encoding=json.dumps(face_encoding)
        )
        
        db.session.add(student)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Student registered successfully',
            'student': {
                'id': student.id,
                'name': student.name,
                'roll_number': student.roll_number
            }
        })
    
    return jsonify({'success': False, 'message': 'Invalid file format'})

@app.route('/api/attendance/report', methods=['GET'])
def get_attendance_report():
    """Get attendance report for students"""
    lecture_id = request.args.get('lecture_id')
    date_str = request.args.get('date')
    
    # Get lecture details if lecture_id is provided
    lecture = None
    if lecture_id:
        lecture = Lecture.query.get(lecture_id)
    
    # Build query based on filters
    query = db.session.query(
        Student,
        Attendance
    ).outerjoin(
        Attendance, 
        (Student.id == Attendance.student_id) & 
        (Attendance.lecture_id == lecture_id if lecture_id else True) &
        (Attendance.date == datetime.strptime(date_str, '%Y-%m-%d').date() if date_str else True)
    )
    
    results = query.all()
    
    # Format attendance data
    attendance = []
    for student, attendance_record in results:
        status = 'present' if attendance_record and attendance_record.status == 'present' else 'absent'
        attendance.append({
            'student': {
                'id': student.id,
                'name': student.name,
                'roll_number': student.roll_number
            },
            'status': status,
            'timestamp': attendance_record.timestamp.isoformat() if attendance_record and attendance_record.timestamp else None
        })
    
    return jsonify({
        'success': True,
        'attendance': attendance,
        'lecture': {
            'id': lecture.id,
            'name': lecture.name,
            'course_code': lecture.course_code,
            'date': lecture.date.strftime('%Y-%m-%d')
        } if lecture else None,
        'total_students': len(attendance)
    })

@app.route('/api/analytics/attendance-trends', methods=['GET'])
def get_attendance_trends():
    """Get attendance trends for analytics"""
    days = int(request.args.get('days', 7))
    lecture_id = request.args.get('lecture_id')
    
    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    # Query attendance data within date range
    query = db.session.query(
        Attendance.date,
        db.func.count(Attendance.id).label('total'),
        db.func.sum(db.case([(Attendance.status == 'present', 1)], else_=0)).label('present')
    )
    
    if lecture_id:
        query = query.filter(Attendance.lecture_id == lecture_id)
    
    query = query.filter(Attendance.date.between(start_date, end_date))
    query = query.group_by(Attendance.date)
    query = query.order_by(Attendance.date)
    
    results = query.all()
    
    # Format data for chart
    dates = []
    attendance_rates = []
    
    for date, total, present in results:
        dates.append(date.strftime('%Y-%m-%d'))
        rate = (present / total * 100) if total > 0 else 0
        attendance_rates.append(round(rate, 1))
    
    # If we have less than the requested days, fill in missing dates with estimated values
    if len(dates) < days:
        # Use simple linear interpolation for missing dates
        all_dates = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
        all_rates = []
        
        for date in all_dates:
            if date in dates:
                idx = dates.index(date)
                all_rates.append(attendance_rates[idx])
            else:
                # Use average if we can't interpolate
                all_rates.append(sum(attendance_rates) / len(attendance_rates) if attendance_rates else 75)
        
        dates = all_dates
        attendance_rates = all_rates
    
    return jsonify({
        'success': True,
        'dates': dates,
        'attendance_rates': attendance_rates
    })

@app.route('/api/analytics/predict-attendance', methods=['GET'])
def predict_attendance():
    """Predict future attendance based on historical data"""
    student_id = request.args.get('student_id')
    days_ahead = int(request.args.get('days_ahead', 7))
    
    # Get historical attendance data
    today = datetime.now().date()
    past_days = 30  # Use last 30 days for prediction
    
    if student_id:
        # Predict for specific student
        student = Student.query.get(student_id)
        if not student:
            return jsonify({'success': False, 'message': 'Student not found'})
            
        # Get attendance history
        attendance_history = Attendance.query.filter(
            Attendance.student_id == student_id,
            Attendance.date >= (today - timedelta(days=past_days))
        ).order_by(Attendance.date).all()
        
        # If we don't have enough data, return a default prediction
        if len(attendance_history) < 5:
            return jsonify({
                'success': True,
                'prediction': {
                    'attendance_probability': 80,  # Default probability
                    'confidence': 'low',
                    'message': 'Not enough historical data for accurate prediction'
                }
            })
        
        # Simple prediction based on recent attendance pattern
        recent_attendance = [1 if a.status == 'present' else 0 for a in attendance_history[-5:]]
        attendance_rate = sum(recent_attendance) / len(recent_attendance) * 100
        
        # Apply a simple trend analysis
        trend = 0
        if len(recent_attendance) >= 3:
            first_half = sum(recent_attendance[:len(recent_attendance)//2])
            second_half = sum(recent_attendance[len(recent_attendance)//2:])
            trend = second_half - first_half
        
        predicted_rate = min(100, max(0, attendance_rate + trend * 10))
        
        return jsonify({
            'success': True,
            'prediction': {
                'student_name': student.name,
                'student_roll': student.roll_number,
                'attendance_probability': round(predicted_rate, 1),
                'trend': 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable',
                'confidence': 'medium'
            }
        })
    else:
        # Predict overall attendance
        # Get all attendance records for the past days
        attendance_records = db.session.query(
            Attendance.date,
            db.func.count(Attendance.id).label('total'),
            db.func.sum(db.case([(Attendance.status == 'present', 1)], else_=0)).label('present')
        ).filter(
            Attendance.date >= (today - timedelta(days=past_days))
        ).group_by(Attendance.date).order_by(Attendance.date).all()
        
        # If we don't have enough data, return a default prediction
        if len(attendance_records) < 5:
            return jsonify({
                'success': True,
                'prediction': {
                    'next_week_rate': 85,  # Default rate
                    'confidence': 'low',
                    'message': 'Not enough historical data for accurate prediction'
                }
            })
        
        # Calculate attendance rates
        dates = []
        rates = []
        
        for date, total, present in attendance_records:
            dates.append((date - today).days)  # Days relative to today
            rate = (present / total * 100) if total > 0 else 0
            rates.append(rate)
        
        # Simple linear regression for prediction
        X = np.array(dates).reshape(-1, 1)
        y = np.array(rates)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict for future days
        future_days = np.array([i+1 for i in range(days_ahead)]).reshape(-1, 1)
        predictions = model.predict(future_days)
        
        # Ensure predictions are within valid range
        predictions = np.clip(predictions, 0, 100)
        
        # Get at-risk students (those with < 75% attendance)
        at_risk_students = []
        students = Student.query.all()
        
        for student in students:
            attendance_count = Attendance.query.filter(
                Attendance.student_id == student.id,
                Attendance.date >= (today - timedelta(days=past_days)),
                Attendance.status == 'present'
            ).count()
            
            total_lectures = Attendance.query.filter(
                Attendance.student_id == student.id,
                Attendance.date >= (today - timedelta(days=past_days))
            ).count()
            
            if total_lectures > 0:
                attendance_rate = (attendance_count / total_lectures) * 100
                if attendance_rate < 75:
                    at_risk_students.append({
                        'id': student.id,
                        'name': student.name,
                        'roll_number': student.roll_number,
                        'attendance_rate': round(attendance_rate, 1)
                    })
        
        return jsonify({
            'success': True,
            'prediction': {
                'next_week_rate': round(float(predictions[6]), 1),  # Prediction for 7th day
                'daily_predictions': [round(float(p), 1) for p in predictions],
                'trend': 'increasing' if model.coef_[0] > 0 else 'decreasing' if model.coef_[0] < 0 else 'stable',
                'confidence': 'medium',
                'at_risk_students': at_risk_students[:5]  # Limit to top 5
            }
        })

@app.route('/api/test/face-recognition', methods=['POST'])
def test_face_recognition():
    """Test face recognition system with an image and return detailed metrics"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'})
    
    image_file = request.files['image']
    if not image_file or not allowed_file(image_file.filename):
        return jsonify({'success': False, 'message': 'Invalid image format'})
    
    # Get threshold from request or use default
    threshold = float(request.form.get('threshold', 0.5))
    
    # Save the uploaded image temporarily
    temp_filename = f"temp_test_{int(time.time())}.jpg"
    temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
    image_file.save(temp_filepath)
    
    # Start timing
    start_time = time.time()
    
    try:
        # Initialize face recognition system with the specified threshold
        face_system = FaceRecognitionSystem(confidence_threshold=threshold)
        
        # Load known faces from database
        students = Student.query.filter(Student.face_encoding.isnot(None)).all()
        face_system.load_known_faces_from_db(students)
        
        # Detect faces in the image
        image = cv2.imread(temp_filepath)
        if image is None:
            return jsonify({'success': False, 'message': 'Could not read image'})
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_system.face_detector.detectMultiScale(gray, 1.1, 5)
        faces_detected = len(faces)
        
        # Process each face
        recognized_faces = []
        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            
            # Extract features
            face_encoding = face_system._extract_face_features(face_img)
            
            # Compare with known faces
            if len(face_system.known_face_encodings) > 0:
                # Calculate similarities
                similarities = [
                    cosine_similarity([face_encoding], [np.array(enc)])[0][0]
                    for enc in face_system.known_face_encodings
                ]
                
                # Find best match
                best_match_idx = np.argmax(similarities)
                confidence = similarities[best_match_idx]
                
                if confidence >= threshold:
                    student_id = face_system.known_face_ids[best_match_idx]
                    student = Student.query.get(student_id)
                    
                    recognized_faces.append({
                        'student': {
                            'id': student.id,
                            'name': student.name,
                            'roll_number': student.roll_number
                        },
                        'confidence': float(confidence),
                        'position': {
                            'x': int(x),
                            'y': int(y),
                            'width': int(w),
                            'height': int(h)
                        }
                    })
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Calculate accuracy (for testing purposes)
        accuracy = len(recognized_faces) / faces_detected if faces_detected > 0 else 0
        
        return jsonify({
            'success': True,
            'faces_detected': faces_detected,
            'recognized_faces': recognized_faces,
            'processing_time': round(processing_time, 3),
            'accuracy': round(accuracy, 2),
            'threshold_used': threshold
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error during face recognition: {str(e)}'
        })
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

@app.route('/api/notifications/send-absence-alerts', methods=['POST'])
def send_absence_alerts():
    """Send email notifications to absent students"""
    data = request.json
    lecture_id = data.get('lecture_id')
    date_str = data.get('date')
    
    if not lecture_id or not date_str:
        return jsonify({'success': False, 'message': 'Lecture ID and date are required'})
    
    try:
        # Parse date
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        # Get lecture
        lecture = Lecture.query.get(lecture_id)
        if not lecture:
            return jsonify({'success': False, 'message': 'Lecture not found'})
        
        # Get all students
        students = Student.query.all()
        student_ids = [s.id for s in students]
        
        # Get attendance records for the specified date and lecture
        attendances = Attendance.query.filter_by(
            lecture_id=lecture_id,
            date=date
        ).all()
        
        # Find absent students
        present_student_ids = [a.student_id for a in attendances if a.status == 'present']
        absent_student_ids = [sid for sid in student_ids if sid not in present_student_ids]
        
        # Get absent student details
        absent_students = Student.query.filter(Student.id.in_(absent_student_ids)).all()
        
        # Email configuration (would be used in a real implementation)
        # sender_email = "attendance@example.com"
        # password = "your_password"  # In production, use environment variables
        # smtp_server = "smtp.example.com"
        # port = 587
        
        # Prepare email content
        email_subject = f"Absence Alert: {lecture.name} on {date_str}"
        
        notifications = []
        for student in absent_students:
            email_body = f"""
            Dear {student.name},
            
            Our records indicate that you were absent from the following class:
            
            Course: {lecture.course_code}
            Lecture: {lecture.name}
            Date: {date_str}
            
            If you believe this is an error, please contact your instructor.
            
            Best regards,
            Attendance System
            """
            
            # In a real implementation, send the actual email
            # with smtplib.SMTP(smtp_server, port) as server:
            #     server.starttls()
            #     server.login(sender_email, password)
            #     
            #     message = MIMEMultipart()
            #     message["From"] = sender_email
            #     message["To"] = student_email  # You would need to store student emails
            #     message["Subject"] = email_subject
            #     
            #     message.attach(MIMEText(email_body, "plain"))
            #     server.sendmail(sender_email, student_email, message.as_string())
            
            notifications.append({
                'student_id': student.id,
                'student_name': student.name,
                'roll_number': student.roll_number,
                'email_subject': email_subject,
                'email_body': email_body
            })
        
        return jsonify({
            'success': True,
            'notifications_count': len(notifications),
            'notifications': notifications
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error sending notifications: {str(e)}'
        })

@app.route('/api/attendance/face-recognition', methods=['POST'])
def mark_attendance_with_face_recognition():
    """Mark attendance using face recognition"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'})
    
    file = request.files['image']
    lecture_id = request.form.get('lecture_id')
    
    if not lecture_id:
        return jsonify({'success': False, 'message': 'Lecture ID is required'})
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No image selected'})
    
    if file and allowed_file(file.filename):
        # Save the uploaded image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attendance_{timestamp}_capture.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Initialize face recognition system
        face_system = FaceRecognitionSystem()
        
        # Load known faces from database
        students = Student.query.all()
        face_system.load_known_faces_from_db(students)
        
        # Recognize faces in the image
        recognized_student_ids = face_system.recognize_faces(filepath)
        
        # Mark attendance for recognized students
        attendance_date = datetime.now().date()
        attendance_records = []
        
        for student_id in recognized_student_ids:
            # Check if attendance already marked
            existing = Attendance.query.filter_by(
                student_id=student_id,
                lecture_id=lecture_id,
                date=attendance_date
            ).first()
            
            if not existing:
                # Create new attendance record
                attendance = Attendance(
                    student_id=student_id,
                    lecture_id=lecture_id,
                    date=attendance_date,
                    status='present'
                )
                db.session.add(attendance)
                
                # Get student details for response
                student = Student.query.get(student_id)
                attendance_records.append({
                    'id': student_id,
                    'name': student.name,
                    'roll_number': student.roll_number
                })
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Marked attendance for {len(attendance_records)} students',
            'recognized_students': attendance_records,
            'image_path': filename
        })
    
    return jsonify({'success': False, 'message': 'Invalid file format'})

@app.route('/api/students', methods=['GET'])
def get_students():
    students = Student.query.all()
    return jsonify({
        'students': [
            {
                'id': student.id,
                'name': student.name,
                'roll_number': student.roll_number,
                'has_face_encoding': student.face_encoding is not None
            } for student in students
        ]
    })

@app.route('/api/students', methods=['POST'])
def add_student():
    data = request.form
    name = data.get('name')
    roll_number = data.get('roll_number')
    
    if not name or not roll_number:
        return jsonify({'success': False, 'message': 'Name and roll number are required'})
    
    # Check if student already exists
    existing_student = Student.query.filter_by(roll_number=roll_number).first()
    if existing_student:
        return jsonify({'success': False, 'message': 'Student with this roll number already exists'})
    
    student = Student(name=name, roll_number=roll_number)
    
    # Process face image if provided
    if 'face_image' in request.files:
        face_image = request.files['face_image']
        if face_image and allowed_file(face_image.filename):
            filename = secure_filename(f"{roll_number}_{face_image.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            face_image.save(filepath)
            
            face_encoding = get_face_encoding(filepath)
            if face_encoding:
                student.face_encoding = json.dumps(face_encoding)
            else:
                return jsonify({'success': False, 'message': 'No face detected in the image'})
    
    db.session.add(student)
    db.session.commit()
    
    return jsonify({'success': True, 'student_id': student.id})

@app.route('/api/students/<int:student_id>/face', methods=['POST'])
def update_student_face(student_id):
    student = Student.query.get_or_404(student_id)
    
    if 'face_image' not in request.files:
        return jsonify({'success': False, 'message': 'No face image provided'})
    
    face_image = request.files['face_image']
    if face_image and allowed_file(face_image.filename):
        filename = secure_filename(f"{student.roll_number}_{face_image.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        face_image.save(filepath)
        
        face_encoding = get_face_encoding(filepath)
        if face_encoding:
            student.face_encoding = json.dumps(face_encoding)
            db.session.commit()
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'No face detected in the image'})
    
    return jsonify({'success': False, 'message': 'Invalid image file'})

@app.route('/api/attendance/take', methods=['POST'])
def take_attendance():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'})
    
    lecture_id = request.form.get('lecture_id')
    if not lecture_id:
        return jsonify({'success': False, 'message': 'Lecture ID is required'})
    
    image_file = request.files['image']
    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image_file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)
        
        # Process the image for face recognition using our face recognition system
        face_system = FaceRecognitionSystem()
        
        # Load known faces from database
        students = Student.query.filter(Student.face_encoding.isnot(None)).all()
        face_system.load_known_faces_from_db(students)
        
        # Recognize faces in the image
        recognized_ids = face_system.recognize_faces(filepath)
        
        if not recognized_ids:
            return jsonify({'success': False, 'message': 'No faces recognized in the image'})
        
        # Record attendance
        recognized_students = []
        today = datetime.now().date()
        
        for student_id in recognized_ids:
            # Check if attendance already recorded for this student today
            existing_attendance = Attendance.query.filter_by(
                student_id=student_id,
                lecture_id=lecture_id,
                date=today
            ).first()
            
            if existing_attendance:
                existing_attendance.status = 'present'
                existing_attendance.timestamp = datetime.utcnow()
            else:
                # Record new attendance
                attendance = Attendance(
                    student_id=student_id,
                    lecture_id=lecture_id,
                    date=today,
                    status='present'
                )
                db.session.add(attendance)
            
            student = Student.query.get(student_id)
            recognized_students.append({
                'id': student.id,
                'name': student.name,
                'roll_number': student.roll_number
            })
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'recognized_students': recognized_students,
            'total_recognized': len(recognized_students)
        })
    
    return jsonify({'success': False, 'message': 'Invalid image file'})

@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    lecture_id = request.args.get('lecture_id')
    date_str = request.args.get('date')
    
    if not lecture_id:
        return jsonify({'success': False, 'message': 'Lecture ID is required'})
    
    query = Attendance.query.filter_by(lecture_id=lecture_id)
    
    if date_str:
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            query = query.filter_by(date=date)
        except ValueError:
            return jsonify({'success': False, 'message': 'Invalid date format. Use YYYY-MM-DD'})
    
    attendances = query.all()
    
    return jsonify({
        'success': True,
        'attendance': [
            {
                'id': attendance.id,
                'student': {
                    'id': attendance.student.id,
                    'name': attendance.student.name,
                    'roll_number': attendance.student.roll_number
                },
                'lecture_id': attendance.lecture_id,
                'date': attendance.date.strftime('%Y-%m-%d'),
                'status': attendance.status,
                'timestamp': attendance.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            } for attendance in attendances
        ]
    })

@app.route('/api/attendance/manual', methods=['POST'])
def manual_attendance():
    data = request.json
    student_id = data.get('student_id')
    lecture_id = data.get('lecture_id')
    status = data.get('status', 'present')  # Default to present
    date_str = data.get('date')
    
    if not student_id or not lecture_id:
        return jsonify({'success': False, 'message': 'Student ID and Lecture ID are required'})
    
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d').date() if date_str else datetime.now().date()
    except ValueError:
        return jsonify({'success': False, 'message': 'Invalid date format. Use YYYY-MM-DD'})
    
    # Check if student exists
    student = Student.query.get(student_id)
    if not student:
        return jsonify({'success': False, 'message': 'Student not found'})
    
    # Check if attendance already recorded
    existing_attendance = Attendance.query.filter_by(
        student_id=student_id,
        lecture_id=lecture_id,
        date=date
    ).first()
    
    if existing_attendance:
        existing_attendance.status = status
        existing_attendance.timestamp = datetime.utcnow()
    else:
        attendance = Attendance(
            student_id=student_id,
            lecture_id=lecture_id,
            date=date,
            status=status
        )
        db.session.add(attendance)
    
    db.session.commit()
    
    return jsonify({'success': True})

# Initialize the database
with app.app_context():
    db.create_all()
    
    # Add default admin user if not exists
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(
            username='admin',
            password='admin123',  # In production, use proper password hashing
            name='Administrator',
            role='admin'
        )
        db.session.add(admin)
        
        # Add default teacher user
        teacher = User(
            username='Akash',
            password='12345',  # Match the credentials in login.html
            name='Mr. Aakash Tripathi',
            role='teacher'
        )
        db.session.add(teacher)
        
        db.session.commit()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)