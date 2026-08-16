import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List
import time


class HeadPosition(Enum):
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class FacialRatios:
    bizygomatic_width_to_height: float
    facial_thirds_upper: float
    facial_thirds_middle: float
    facial_thirds_lower: float
    intercanthal_to_eye_width: float
    eye_width_to_height: float
    nose_width_to_mouth_width: float
    face_width_to_lower_face_height: float
    mandible_width_to_face_width: float


class FaceRatioAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.cap = None
        self.current_position = HeadPosition.FRONT
        self.captured_landmarks = {HeadPosition.FRONT: None, HeadPosition.LEFT: None, HeadPosition.RIGHT: None}
        
    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
    def stop_webcam(self):
        if self.cap:
            self.cap.release()
            
    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_landmark_point(self, landmarks, index):
        return [landmarks.landmark[index].x * self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                landmarks.landmark[index].y * self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
    
    def calculate_ratios(self, landmarks) -> FacialRatios:
        """Calculate various facial ratios from face landmarks"""
        # Key landmark indices for MediaPipe Face Mesh
        # These are approximate indices based on the 468-point face mesh
        
        # Bizygomatic width (cheekbone width) - approximate
        left_cheek = self.get_landmark_point(landmarks, 234)
        right_cheek = self.get_landmark_point(landmarks, 454)
        bizygomatic_width = self.calculate_distance(left_cheek, right_cheek)
        
        # Facial height (hairline to chin)
        hairline = self.get_landmark_point(landmarks, 10)
        chin = self.get_landmark_point(landmarks, 152)
        facial_height = self.calculate_distance(hairline, chin)
        
        # Facial thirds
        # Upper third: hairline to glabella (between eyebrows)
        glabella = self.get_landmark_point(landmarks, 9)
        upper_third = self.calculate_distance(hairline, glabella)
        
        # Middle third: glabella to subnasale (under nose)
        subnasale = self.get_landmark_point(landmarks, 2)
        middle_third = self.calculate_distance(glabella, subnasale)
        
        # Lower third: subnasale to chin
        lower_third = self.calculate_distance(subnasale, chin)
        
        # Eye measurements
        left_eye_outer = self.get_landmark_point(landmarks, 33)
        left_eye_inner = self.get_landmark_point(landmarks, 133)
        right_eye_outer = self.get_landmark_point(landmarks, 263)
        right_eye_inner = self.get_landmark_point(landmarks, 362)
        
        # Intercanthal distance (distance between inner eye corners)
        intercanthal_distance = self.calculate_distance(left_eye_inner, right_eye_inner)
        
        # Eye width (average of both eyes)
        left_eye_width = self.calculate_distance(left_eye_outer, left_eye_inner)
        right_eye_width = self.calculate_distance(right_eye_outer, right_eye_inner)
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        
        # Eye height (average of both eyes)
        left_eye_top = self.get_landmark_point(landmarks, 159)
        left_eye_bottom = self.get_landmark_point(landmarks, 145)
        right_eye_top = self.get_landmark_point(landmarks, 386)
        right_eye_bottom = self.get_landmark_point(landmarks, 374)
        
        left_eye_height = self.calculate_distance(left_eye_top, left_eye_bottom)
        right_eye_height = self.calculate_distance(right_eye_top, right_eye_bottom)
        avg_eye_height = (left_eye_height + right_eye_height) / 2
        
        # Nose width
        left_nose = self.get_landmark_point(landmarks, 129)
        right_nose = self.get_landmark_point(landmarks, 358)
        nose_width = self.calculate_distance(left_nose, right_nose)
        
        # Mouth width
        left_mouth = self.get_landmark_point(landmarks, 61)
        right_mouth = self.get_landmark_point(landmarks, 291)
        mouth_width = self.calculate_distance(left_mouth, right_mouth)
        
        # Lower face height (subnasale to chin)
        lower_face_height = self.calculate_distance(subnasale, chin)
        
        # Mandible width (jaw width at widest point)
        left_jaw = self.get_landmark_point(landmarks, 172)
        right_jaw = self.get_landmark_point(landmarks, 397)
        mandible_width = self.calculate_distance(left_jaw, right_jaw)
        
        # Calculate ratios
        ratios = FacialRatios(
            bizygomatic_width_to_height=bizygomatic_width / facial_height if facial_height > 0 else 0,
            facial_thirds_upper=upper_third / facial_height if facial_height > 0 else 0,
            facial_thirds_middle=middle_third / facial_height if facial_height > 0 else 0,
            facial_thirds_lower=lower_third / facial_height if facial_height > 0 else 0,
            intercanthal_to_eye_width=intercanthal_distance / avg_eye_width if avg_eye_width > 0 else 0,
            eye_width_to_height=avg_eye_width / avg_eye_height if avg_eye_height > 0 else 0,
            nose_width_to_mouth_width=nose_width / mouth_width if mouth_width > 0 else 0,
            face_width_to_lower_face_height=bizygomatic_width / lower_face_height if lower_face_height > 0 else 0,
            mandible_width_to_face_width=mandible_width / bizygomatic_width if bizygomatic_width > 0 else 0
        )
        
        return ratios
    
    def detect_head_turn(self, landmarks) -> Optional[HeadPosition]:
        """Detect if user is turning head left or right based on landmark visibility"""
        # Check visibility of key landmarks on each side
        left_eye_visible = landmarks.landmark[33].visibility > 0.5
        right_eye_visible = landmarks.landmark[263].visibility > 0.5
        
        # Check nose tip position relative to face center
        nose_tip = landmarks.landmark[1]
        face_center_x = 0.5
        
        if not left_eye_visible and right_eye_visible:
            return HeadPosition.LEFT
        elif not right_eye_visible and left_eye_visible:
            return HeadPosition.RIGHT
        elif abs(nose_tip.x - face_center_x) < 0.1:
            return HeadPosition.FRONT
        elif nose_tip.x < face_center_x - 0.1:
            return HeadPosition.LEFT
        elif nose_tip.x > face_center_x + 0.1:
            return HeadPosition.RIGHT
        
        return None
    
    def draw_instructions(self, frame, position: HeadPosition):
        """Draw instructions on the frame"""
        height, width = frame.shape[:2]
        
        # Create a semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        instructions = {
            HeadPosition.FRONT: "Please face the camera directly",
            HeadPosition.LEFT: "Please turn your head to the LEFT",
            HeadPosition.RIGHT: "Please turn your head to the RIGHT"
        }
        
        cv2.putText(frame, instructions[position], (50, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show progress
        captured = [k for k, v in self.captured_landmarks.items() if v is not None]
        progress_text = f"Progress: {len(captured)}/3 positions captured"
        cv2.putText(frame, progress_text, (50, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show which positions are captured
        y_offset = 130
        for pos in [HeadPosition.FRONT, HeadPosition.LEFT, HeadPosition.RIGHT]:
            status = "✓" if self.captured_landmarks[pos] is not None else "○"
            color = (0, 255, 0) if self.captured_landmarks[pos] is not None else (255, 255, 255)
            cv2.putText(frame, f"{status} {pos.value.upper()}", (50, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
    
    def run(self):
        """Main application loop"""
        self.start_webcam()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Draw face mesh
                        self.mp_drawing.draw_landmarks(
                            frame, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION,
                            self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                            self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                        )
                        
                        # Detect head position
                        detected_position = self.detect_head_turn(face_landmarks)
                        
                        if detected_position:
                            # Store landmarks for current position if not already captured
                            if self.captured_landmarks[detected_position] is None:
                                self.captured_landmarks[detected_position] = face_landmarks
                                print(f"Captured {detected_position.value} position")
                            
                            # Move to next position if current is captured
                            if self.captured_landmarks[self.current_position] is not None:
                                if self.current_position == HeadPosition.FRONT:
                                    self.current_position = HeadPosition.LEFT
                                elif self.current_position == HeadPosition.LEFT:
                                    self.current_position = HeadPosition.RIGHT
                                elif self.current_position == HeadPosition.RIGHT:
                                    # All positions captured, calculate and display results
                                    self.display_results()
                                    return
                        
                        self.draw_instructions(frame, self.current_position)
                
                else:
                    cv2.putText(frame, "No face detected. Please position your face in the frame.", 
                               (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow('Face Ratio Analyzer', frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.stop_webcam()
            cv2.destroyAllWindows()
    
    def display_results(self):
        """Calculate and display facial ratios"""
        if self.captured_landmarks[HeadPosition.FRONT] is None:
            print("Error: Front view not captured")
            return
        
        front_landmarks = self.captured_landmarks[HeadPosition.FRONT]
        ratios = self.calculate_ratios(front_landmarks)
        
        print("\n" + "="*60)
        print("FACIAL RATIO ANALYSIS RESULTS")
        print("="*60)
        print(f"\nBizygomatic Width / Facial Height: {ratios.bizygomatic_width_to_height:.3f}")
        print(f"  (Ideal: ~0.7-0.8)")
        
        print(f"\nFacial Thirds (as proportion of total face height):")
        print(f"  Upper third (hairline to glabella): {ratios.facial_thirds_upper:.3f}")
        print(f"  Middle third (glabella to subnasale): {ratios.facial_thirds_middle:.3f}")
        print(f"  Lower third (subnasale to chin): {ratios.facial_thirds_lower:.3f}")
        print(f"  (Ideal: Each ~0.33 for harmonious proportions)")
        
        print(f"\nIntercanthal Distance / Eye Width: {ratios.intercanthal_to_eye_width:.3f}")
        print(f"  (Ideal: ~1.0)")
        
        print(f"\nEye Width / Eye Height: {ratios.eye_width_to_height:.3f}")
        print(f"  (Ideal: ~2.0-3.0)")
        
        print(f"\nNose Width / Mouth Width: {ratios.nose_width_to_mouth_width:.3f}")
        print(f"  (Ideal: ~0.5-0.6)")
        
        print(f"\nFace Width / Lower Face Height: {ratios.face_width_to_lower_face_height:.3f}")
        print(f"  (Ideal: ~1.2-1.4)")
        
        print(f"\nMandible Width / Face Width: {ratios.mandible_width_to_face_width:.3f}")
        print(f"  (Ideal: ~0.8-0.9)")
        
        print("\n" + "="*60)
        print("Analysis complete!")
        print("="*60 + "\n")
        
        # Keep window open briefly
        time.sleep(5)


if __name__ == "__main__":
    analyzer = FaceRatioAnalyzer()
    analyzer.run()
