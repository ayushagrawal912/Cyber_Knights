import cv2
import numpy as np
import os
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognitionSystem:
    def __init__(self, known_faces_dir='known_faces', confidence_threshold=0.5):
        self.known_faces_dir = known_faces_dir
        self.confidence_threshold = confidence_threshold
        self.known_face_encodings = []
        self.known_face_ids = []
        
        # Load face detection model
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # We'll use HOG features for face recognition
        self.use_simple_features = True
            
        # Ensure directory exists
        os.makedirs(known_faces_dir, exist_ok=True)
    
    def load_known_faces_from_db(self, students):
        """Load face encodings from database students"""
        self.known_face_encodings = []
        self.known_face_ids = []
        
        for student in students:
            if student.face_encoding:
                try:
                    face_encoding = np.array(json.loads(student.face_encoding))
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_ids.append(student.id)
                except (json.JSONDecodeError, TypeError):
                    print(f"Error loading face encoding for student {student.id}")
        
        return len(self.known_face_encodings)
    
    def _extract_face_features(self, face_img):
        """Extract features from face using HOG"""
        # Resize to standard size
        face_img = cv2.resize(face_img, (100, 100))
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate HOG features
        hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        hog_features = hog.compute(cv2.resize(gray, (64, 64)))
        
        # Use pixel values as additional features
        resized_gray = cv2.resize(gray, (32, 32)).flatten() / 255.0
        
        # Combine features
        features = np.concatenate([hog_features.flatten(), resized_gray])
        
        # Normalize
        if np.linalg.norm(features) > 0:
            features = features / np.linalg.norm(features)
            
        return features
    
    def encode_face_image(self, image_path):
        """Encode a face from an image file"""
        if not os.path.exists(image_path):
            return None
            
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            return None
        
        # Get the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        face_img = image[y:y+h, x:x+w]
        
        # Extract features
        face_encoding = self._extract_face_features(face_img)
            
        return face_encoding.tolist()
    
    def recognize_faces(self, image_path):
        """Recognize faces in an image and return student IDs"""
        if not os.path.exists(image_path):
            return []
            
        image = cv2.imread(image_path)
        if image is None:
            return []
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 5)
        
        recognized_ids = []
        
        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            
            # Extract features
            if self.use_simple_features:
                face_encoding = self._extract_face_features(face_img)
            else:
                # Use OpenCV DNN face recognition
                face_encoding = self.face_recognizer.feature(face_img)
            
            # Compare with known faces
            if len(self.known_face_encodings) > 0:
                # Calculate similarities
                similarities = [cosine_similarity(
                    np.array(face_encoding).reshape(1, -1), 
                    np.array(known_encoding).reshape(1, -1)
                )[0][0] for known_encoding in self.known_face_encodings]
                
                best_match_index = np.argmax(similarities)
                
                if similarities[best_match_index] > self.confidence_threshold:
                    student_id = self.known_face_ids[best_match_index]
                    recognized_ids.append(student_id)
        
        return recognized_ids
    
    def recognize_faces_from_webcam(self, camera_index=0, max_frames=30):
        """Capture image from webcam and recognize faces"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            return {"success": False, "message": "Failed to open webcam"}
        
        # Capture a few frames to allow camera to adjust
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return {"success": False, "message": "Failed to capture image from webcam"}
        
        # Capture the actual frame for recognition
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return {"success": False, "message": "Failed to capture image from webcam"}
        
        # Save the captured frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_path = f"captures/capture_{timestamp}.jpg"
        os.makedirs("captures", exist_ok=True)
        cv2.imwrite(capture_path, frame)
        
        # Recognize faces in the captured image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        recognized_ids = []
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=self.tolerance
            )
            
            if True in matches:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    student_id = self.known_face_ids[best_match_index]
                    recognized_ids.append(student_id)
        
        return {
            "success": True,
            "recognized_ids": recognized_ids,
            "total_faces": len(face_locations),
            "capture_path": capture_path
        }
    
    def mark_faces_in_image(self, image_path, output_path=None):
        """Mark recognized faces in an image and save the result"""
        if not os.path.exists(image_path):
            return None
            
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # Convert image to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=self.tolerance
            )
            
            name = "Unknown"
            
            if True in matches:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    student_id = self.known_face_ids[best_match_index]
                    name = f"ID: {student_id}"
            
            # Draw a rectangle around the face
            cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw a label with the name below the face
            cv2.rectangle(image_bgr, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image_bgr, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image_bgr)
            return output_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"marked/marked_{timestamp}.jpg"
            os.makedirs("marked", exist_ok=True)
            cv2.imwrite(output_path, image_bgr)
            return output_path