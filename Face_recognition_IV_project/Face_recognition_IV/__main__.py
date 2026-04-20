#!/usr/bin/env python3
import cv2
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime

def check_libraries():
    """Check if all required libraries are installed"""
    print("🔍 Checking required libraries...")
    
    try:
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not found. Install with: pip install opencv-python")
        return False
    
    try:
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy not found. Install with: pip install numpy")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL/Pillow: Available")
    except ImportError:
        print("❌ PIL not found. Install with: pip install Pillow")
        return False
    
    print("✅ All libraries are available!")
    return True

class FaceRecognitionSystem:
    def __init__(self):
        self.data_dir = "data"
        self.model_path = "face_recognizer_model.xml"
        self.users_config = "users.json"
        self.screenshots_dir = "screenshots"
        
        self.face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        self.recognizer = None

        self.create_directories()
        
        self.users = self.load_users_config()
        
        print("🚀 Face Recognition System initialized!")

    def create_directories(self):
        directories = [self.data_dir, self.screenshots_dir]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"📁 Created directory: {directory}")
    
    def load_users_config(self):
        if os.path.exists(self.users_config):
            try:
                with open(self.users_config, 'r') as f:
                    users = json.load(f)
                print(f"📋 Loaded {len(users)} users from configuration")
                return users
            except Exception as e:
                print(f"⚠️ Error loading users config: {e}")
        

        default_users = {"1": "User1"}
        self.save_users_config(default_users)
        print("📋 Created default users configuration")
        return default_users
    
    def save_users_config(self, users=None):
        if users is None:
            users = self.users
        
        try:
            with open(self.users_config, 'w') as f:
                json.dump(users, f, indent=2)
            print("💾 Users configuration saved")
        except Exception as e:
            print(f"❌ Error saving users config: {e}")
    
# ============================================================================
# STEP 3: USER MANAGEMENT
# ============================================================================

    def add_user(self, user_id, name):
        try:
            user_id = str(user_id)
            self.users[user_id] = name
            self.save_users_config()
            print(f"✅ Added user: {name} (ID: {user_id})")
            return True
        except Exception as e:
            print(f"❌ Error adding user: {e}")
            return False
    
    def list_users(self):
        print("\n👥 REGISTERED USERS:")
        print("-" * 30)
        if len(self.users) == 0:
            print("No users registered")
        else:
            for user_id, name in self.users.items():
                print(f"  ID: {user_id} - Name: {name}")
        print()
    
    def remove_user(self, user_id):
        user_id = str(user_id)
        if user_id in self.users:
            name = self.users[user_id]
            del self.users[user_id]
            self.save_users_config()
            print(f"✅ Removed user: {name} (ID: {user_id})")
            return True
        else:
            print(f"❌ User ID {user_id} not found")
            return False

# ============================================================================
# STEP 4: FACE DETECTION
# ============================================================================

    def detect_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Return the largest face if multiple detected
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        
        x, y, w, h = faces[0]
        cropped_face = img[y:y+h, x:x+w]
        return cropped_face
    
    def test_camera(self):
        print("📹 Testing camera...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cannot access camera")
            return False
        
        print("✅ Camera accessible")
        print("📸 Press any key to take test photo, 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Cannot read frame")
                break
            
            # Add text overlay
            cv2.putText(frame, "Camera Test - Press any key", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Camera Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key != 255:  # Any other key pressed
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"camera_test_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Test photo saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        return True

# ============================================================================
# STEP 5: DATA COLLECTION FOR TRAINING
# ============================================================================
    def collect_training_data(self, user_id, num_samples=200):
        print(f"📸 Starting data collection for User ID: {user_id}")
        print(f"🎯 Target samples: {num_samples}")
        print("\n💡 Instructions:")
        print("  - Look directly at the camera")
        print("  - Keep your face clearly visible")
        print("  - Move your head slightly for different angles")
        print("  - Ensure good lighting")
        print("  - Press 'q' to quit early")
        if str(user_id) not in self.users:
            print(f"❌ User ID {user_id} not found. Please add user first.")
            return False
        
        user_name = self.users[str(user_id)]
        print(f"👤 Collecting data for: {user_name}")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cannot access camera")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        img_count = 0
        print("\n🎬 Data collection started!")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Cannot read frame")
                break

            # Create display frame
            display_frame = frame.copy()
            
            # Add information overlay
            cv2.putText(display_frame, f"User: {user_name} (ID: {user_id})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Samples: {img_count}/{num_samples}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'q' to quit", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show original frame
            cv2.imshow("📸 Data Collection - Camera Feed", display_frame)

            # Detect and process face
            cropped_face = self.detect_face(frame)
            
            if cropped_face is not None:
                img_count += 1
                
                # Resize to standard size
                face_resized = cv2.resize(cropped_face, (200, 200))
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                
                # Save image
                filename = f"{self.data_dir}/user.{user_id}.{img_count}.jpg"
                cv2.imwrite(filename, face_gray)
                
                # Show processed face
                display_face = face_gray.copy()
                cv2.putText(display_face, f"Sample: {img_count}/{num_samples}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow("✅ Processed Face", display_face)

                # Progress update
                if img_count % 20 == 0:
                    print(f"📊 Progress: {img_count}/{num_samples} samples collected")
                
                # Stop when target reached
                if img_count >= num_samples:
                    print(f"🎉 Target reached! Collected {img_count} samples")
                    break
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"⏹️ Collection stopped by user at {img_count} samples")
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        if img_count > 0:
            print(f"✅ Data collection completed!")
            print(f"📊 Collected {img_count} samples for {user_name}")
            print(f"📁 Files saved in '{self.data_dir}' directory")
            return True
        else:
            print("❌ No samples collected")
            return False
        
# ============================================================================
# STEP 6: MODEL TRAINING
# ============================================================================

    def train_model(self):
        print("🧠 Starting model training...")
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            print(f"❌ Data directory '{self.data_dir}' not found")
            print("💡 Collect training data first")
            return False
        
        # Get all training images
        image_files = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg')]
        
        if len(image_files) == 0:
            print("❌ No training images found")
            print("💡 Collect training data first")
            return False
        
        print(f"📊 Found {len(image_files)} training images")

        # Prepare training data
        faces = []
        labels = []
        
        print("📝 Processing training images...")
        
        successful_images = 0
        failed_images = 0

        for image_file in image_files:
            try:
                # Load image
                image_path = os.path.join(self.data_dir, image_file)
                img = Image.open(image_path).convert('L')  # Convert to grayscale
                
                # Convert to numpy array
                face_np = np.array(img, 'uint8')
                
                # Extract user ID from filename (user.ID.number.jpg)
                parts = image_file.split('.')
                if len(parts) >= 3:
                    user_id = int(parts[1])
                else:
                    print(f"⚠️ Skipping {image_file}: Invalid filename format")
                    failed_images += 1
                    continue
                # Add to training data
                faces.append(face_np)
                labels.append(user_id)
                successful_images += 1
                
                # Show progress
                if successful_images % 50 == 0:
                    print(f"   📊 Processed {successful_images}/{len(image_files)} images")
                
            except Exception as e:
                print(f"⚠️ Error processing {image_file}: {e}")
                failed_images += 1
                continue
            
        if successful_images == 0:
            print("❌ No valid training images found")
            return False
        
        print(f"✅ Successfully processed {successful_images} images")
        if failed_images > 0:
            print(f"⚠️ Failed to process {failed_images} images")
        
        # Convert to numpy arrays
        faces = np.array(faces)
        labels = np.array(labels)

        # Show training summary
        unique_users = len(set(labels))
        print(f"👥 Training data for {unique_users} different users")

        # Count samples per user
        user_counts = {}
        for label in labels:
            user_counts[label] = user_counts.get(label, 0) + 1
        
        print("📊 Samples per user:")
        for user_id in sorted(user_counts.keys()):
            count = user_counts[user_id]
            name = self.users.get(str(user_id), f"User {user_id}")
            print(f"   👤 {name} (ID: {user_id}): {count} samples")
        
        # Create and train the recognizer
        print("🔄 Training LBPH Face Recognizer...")
        
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.train(faces, labels)

        # Save the trained model
        self.recognizer.write(self.model_path)
        
        print(f"✅ Model training completed successfully!")
        print(f"💾 Model saved as: {self.model_path}")
        
        # Show model file info
        model_size = os.path.getsize(self.model_path) / 1024  # KB
        print(f"📊 Model file size: {model_size:.1f} KB")
        
        return True

# ============================================================================
# STEP 7: FACE RECOGNITION
# ============================================================================

    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            print(f"❌ Model file '{self.model_path}' not found")
            print("💡 Train the model first")
            return False
        
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(self.model_path)
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def start_recognition(self, confidence_threshold=75):
        """
        Start real-time face recognition
        """
        print(f"🔍 Starting face recognition...")
        print(f"🎯 Confidence threshold: {confidence_threshold}%")
        print("📹 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save screenshot")
        print("   - Press 'c' to change confidence threshold")

        if self.recognizer is None:
            if not self.load_model():
                return False
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cannot access camera")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera opened successfully")
        print("\n🎥 Face recognition started!")
        
        screenshot_count = 0
        frame_count = 0
        recognition_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Cannot read frame")
                    break
                
                frame_count += 1
                display_frame = frame.copy()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = self.face_classifier.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(50, 50)
                )

                for (x, y, w, h) in faces:
                    # Extract face region
                    face_roi = gray[y:y + h, x:x + w]

                    try:
                        face_resized = cv2.resize(face_roi, (200, 200))
                        user_id, confidence_score = self.recognizer.predict(face_resized)
                        confidence = int(100 * (1 - confidence_score / 300))
                        confidence = max(0, min(100, confidence))
                        if confidence >= confidence_threshold:
                            user_name = self.users.get(str(user_id), f"User {user_id}")
                            color = (0, 255, 0)  # Green
                            label = f"{user_name}"
                            status = f"Confidence: {confidence}%"
                            recognition_count += 1
                        else:
                            color = (0, 0, 255)  # Red
                            label = "UNKNOWN"
                            status = f"Low confidence: {confidence}%"
                        
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                        cv2.rectangle(display_frame, (x, y - 35),
                                     (x + text_width, y), color, -1)
                        cv2.putText(display_frame, label, (x + 5, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        cv2.putText(display_frame, status, (x, y + h + 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    except Exception as e:
                        print(f"⚠️ Error processing face: {e}")
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(display_frame, "ERROR", (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                info_y = 30
                cv2.putText(display_frame, f"Faces detected: {len(faces)}", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                info_y += 25
                cv2.putText(display_frame, f"Confidence threshold: {confidence_threshold}%", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                info_y += 25
                cv2.putText(display_frame, f"Recognitions: {recognition_count}", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                height = display_frame.shape[0]
                cv2.putText(display_frame, "Controls: 'q'=quit, 's'=screenshot, 'c'=change confidence",
                           (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow("🔍 Face Recognition System", display_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n🛑 Quitting face recognition...")
                    break

                if key == ord('q'):
                    print("\n🛑 Quitting face recognition...")
                    break

                elif key == ord('s'):
                    screenshot_count += 1
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{self.screenshots_dir}/recognition_{timestamp}_{screenshot_count}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"📸 Screenshot saved: {filename}")
                
                elif key == ord('c'):
                    print(f"\n🎯 Current confidence threshold: {confidence_threshold}%")
                    try:
                        new_threshold = input("Enter new threshold (50-95): ")
                        new_threshold = int(new_threshold)
                        if 50 <= new_threshold <= 95:
                            confidence_threshold = new_threshold
                            print(f"✅ Confidence threshold changed to: {confidence_threshold}%")
                        else:
                            print("⚠️ Invalid range. Keep current value.")
                    except ValueError:
                        print("⚠️ Invalid input. Keep current value.")
        
        except KeyboardInterrupt:
            print("\n🛑 Recognition interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n📊 RECOGNITION SESSION SUMMARY:")
            print(f"   Total frames processed: {frame_count}")
            print(f"   Successful recognitions: {recognition_count}")
            print(f"   Screenshots saved: {screenshot_count}")
            print("✅ Face recognition session completed!")
        
        return True

# ============================================================================
# STEP 8: SYSTEM STATUS AND UTILITIES
# ============================================================================

    def show_system_status(self):
        print("\n📊 SYSTEM STATUS")
        print("=" * 50)
        print(f"👥 Registered users: {len(self.users)}")
        for user_id, name in self.users.items():
            print(f"   ID {user_id}: {name}")
        if os.path.exists(self.data_dir):
            files = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg')]
            print(f"\n📸 Training images: {len(files)}")
            if len(files) > 0:
                user_counts = {}
                for f in files:
                    try:
                        user_id = f.split('.')[1]
                        user_counts[user_id] = user_counts.get(user_id, 0) + 1
                    except:
                        continue
                
                for user_id, count in user_counts.items():
                    name = self.users.get(user_id, f"User {user_id}")
                    print(f"   👤 {name}: {count} samples")
        else:
            print("\n📸 No training data found")
        
        if os.path.exists(self.model_path):
            model_size = os.path.getsize(self.model_path) / 1024
            model_time = datetime.fromtimestamp(os.path.getmtime(self.model_path))
            print(f"\n🤖 Trained model: {model_size:.1f} KB")
            print(f"📅 Last trained: {model_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"\n🤖 No trained model found")

        print(f"\n📁 Data directory: {self.data_dir}")
        print(f"📁 Screenshots directory: {self.screenshots_dir}")
        print(f"📄 Users config: {self.users_config}")
        print(f"📄 Model file: {self.model_path}")
        
        print("=" * 50)

# ============================================================================
# STEP 9: INTERACTIVE MENU SYSTEM
# ============================================================================

def display_menu():
    print("\n" + "="*60)
    print("🤖 FACE RECOGNITION SYSTEM - MAIN MENU")
    print("="*60)
    print("1. 👤 User Management")
    print("2. 📸 Collect Training Data")
    print("3. 🧠 Train Face Recognition Model")
    print("4. 🔍 Start Face Recognition")
    print("5. 📊 View System Status")
    print("6. 🧪 Test Camera")
    print("7. 🚀 Complete Setup Workflow (Recommended for first time)")
    print("8. ❓ Help & Troubleshooting")
    print("9. 🚪 Exit")
    print("="*60)

def user_management_menu(system):
    while True:
        print("\n👥 USER MANAGEMENT")
        print("-" * 30)
        print("1. 📋 List all users")
        print("2. ➕ Add new user")
        print("3. ❌ Remove user")
        print("4. 🔙 Back to main menu")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            system.list_users()
        
        elif choice == '2':
            try:
                user_id = input("Enter user ID (number): ").strip()
                name = input("Enter user name: ").strip()
                
                if user_id and name:
                    system.add_user(user_id, name)
                else:
                    print("⚠️ Both ID and name are required")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '3':
            system.list_users()
            try:
                user_id = input("Enter user ID to remove: ").strip()
                if user_id:
                    confirm = input(f"Are you sure you want to remove user {user_id}? (y/n): ").lower()
                    if confirm == 'y':
                        system.remove_user(user_id)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '4':
            break
        
        else:
            print("⚠️ Invalid option")

def complete_workflow(system):
    print("\n🚀 COMPLETE SETUP WORKFLOW")
    print("="*50)
    print("This will guide you through the entire setup process:")
    print("1. User management")
    print("2. Data collection")
    print("3. Model training")
    print("4. Testing face recognition")
    print()
    
    input("Press Enter to start the workflow...")
    
    print("\n📝 STEP 1: USER MANAGEMENT")
    print("-" * 40)
    system.list_users()
    
    while True:
        add_user = input("\nDo you want to add a new user? (y/n): ").lower()
        if add_user != 'y':
            break
        
        try:
            user_id = input("Enter user ID (number): ").strip()
            name = input("Enter user name: ").strip()
            
            if user_id and name:
                system.add_user(user_id, name)
            else:
                print("⚠️ Both ID and name are required")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n📸 STEP 2: DATA COLLECTION")
    print("-" * 40)
    
    if len(system.users) == 0:
        print("❌ No users available for data collection")
        print("💡 Add users first")
        return
    
    system.list_users()
    
    for user_id in system.users.keys():
        user_name = system.users[user_id]
        collect = input(f"\nCollect data for {user_name} (ID: {user_id})? (y/n): ").lower()
        
        if collect == 'y':
            try:
                samples = input("Number of samples (default 100): ").strip()
                samples = int(samples) if samples else 100
                
                print(f"\n🎬 Starting data collection for {user_name}")
                print("💡 Position yourself in front of the camera")
                input("Press Enter when ready...")
                
                success = system.collect_training_data(user_id, samples)
                if success:
                    print(f"✅ Data collection completed for {user_name}")
                else:
                    print(f"❌ Data collection failed for {user_name}")
                    
            except ValueError:
                print("⚠️ Invalid number of samples")
            except Exception as e:
                print(f"❌ Error during data collection: {e}")