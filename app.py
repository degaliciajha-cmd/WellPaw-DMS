import os
import cv2
import torch
from flask import Flask, render_template, session, redirect, url_for, flash, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Professional Capstone Tip: Use environment variables for security
app.secret_key = os.environ.get('SECRET_KEY', 'dms_secure_key_123')

# --- DATABASE CONFIGURATION ---
database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

# --- DATABASE CONFIGURATION (XAMPP / MariaDB) ---
# Ensure XAMPP is running Apache and MySQL before starting app.py
# Database name in phpMyAdmin must be exactly: DMS

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/DMS'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- DATABASE MODEL ---
class DogRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    owner_name = db.Column(db.String(100), nullable=False)
    dog_name = db.Column(db.String(100), nullable=False)
    dog_breed = db.Column(db.String(100))
    dog_age = db.Column(db.String(20))
    average_score = db.Column(db.Float)
    stress_level = db.Column(db.String(50))

# --- OPENCV FUNCTION ---
def analyze_dog_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = 0
    if not cap.isOpened():
        return 3.0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
    cap.release()
    return 1.5 if frames > 50 else 2.5

def calculate_status(score):
    if score <= 1.5: return "Excellent"
    elif score <= 2.5: return "Fair"
    elif score <= 3.5: return "Poor"
    else: return "Critical"

# --- NEW FEATURE: API FOR SYNC SCRIPT ---
@app.route('/api/data')
def get_data():
    """Returns raw JSON data for the local sync.py script"""
    records = DogRecord.query.all()
    return jsonify([{
        "id": r.id,
        "owner_name": r.owner_name,
        "dog_name": r.dog_name,
        "dog_breed": r.dog_breed,
        "dog_age": r.dog_age,
        "average_score": r.average_score,
        "stress_level": r.stress_level
    } for r in records])

# --- ROUTES ---
@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('username') == 'admin' and request.form.get('password') == 'admin':
            session['username'] = 'Admin'
            return redirect(url_for('dashboard'))
        flash("Invalid Credentials")
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session: return redirect(url_for('login'))
    records = DogRecord.query.all()
    latest = DogRecord.query.order_by(DogRecord.id.desc()).first()
    return render_template('menu.html', 
                           total_records=len(records), 
                           latest_stress=latest.stress_level if latest else "N/A",
                           username=session.get('username'),
                           rows=records)

@app.route('/monitoring', methods=['GET', 'POST'])
def monitoring():
    if 'username' not in session: 
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Captures the data from your first form to prevent "Unknown" entries
        session['temp_dog'] = {
            'owner_name': request.form.get('owner_name'),
            'dog_name': request.form.get('dog_name'),
            'dog_breed': request.form.get('dog_breed'),
            'dog_age': request.form.get('dog_age')
        }
        return redirect(url_for('video_analysis'))
        
    return render_template('monitoring.html')

# Load the YOLOv5 model globally to prevent NameError
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

@app.route('/video_analysis', methods=['GET', 'POST'])
def video_analysis():
    if request.method == 'POST':
        if 'dog_video' not in request.files:
            flash("No video file detected.")
            return redirect(request.url)
            
        file = request.files['dog_video']
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)

        # 1. Save the video temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 2. Get Owner and Dog names from session for folder naming
        temp_data = session.get('temp_dog', {})
        owner = secure_filename(temp_data.get('owner_name', 'Unknown'))
        dog = secure_filename(temp_data.get('dog_name', 'Unknown'))
        
        # Create a unique folder: static/detections/OwnerName_DogName/
        detection_folder = os.path.join('static/detections', f"{owner}_{dog}")
        os.makedirs(detection_folder, exist_ok=True)

        # 3. Process Video for 5 Valid Detections
        cap = cv2.VideoCapture(filepath)
        detection_count = 0
        frames_processed = 0
        
        while cap.isOpened() and detection_count < 5:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 10th frame to speed up and get variety in the 5 photos
            if frames_processed % 10 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb)
                detections = results.pandas().xyxy[0]
                
                # Check if a dog is in this specific frame
                if not detections[detections['name'] == 'dog'].empty:
                    detection_count += 1
                    # Save the image with boxes to the new folder
                    # Filename example: Owner_Dog_1.jpg
                    img_name = f"{owner}_{dog}_{detection_count}.jpg"
                    save_path = os.path.join(detection_folder, img_name)
                    
                    # results.render() draws the boxes on the image
                    results.render() 
                    # results.ims[0] is the numpy array (BGR) after rendering boxes
                    cv2.imwrite(save_path, cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR))
            
            frames_processed += 1
            
        cap.release()

        # 4. Final Validation
        if detection_count >= 5:
            session['video_score'] = analyze_dog_video(filepath)
            flash(f"Success! 5 valid detections saved for {dog}.")
            return redirect(url_for('questionnaire')) 
        else:
            flash(f"Only found {detection_count} clear views of the dog. Need 5 for valid results.")
            return redirect(url_for('video_analysis'))

    return render_template('video.html')

@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if 'username' not in session: return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            physical_scores = [
                int(request.form.get('weight')), 
                int(request.form.get('skin')), 
                int(request.form.get('fur')), 
                int(request.form.get('parasites')), 
                int(request.form.get('behavior'))
            ]
            physical_avg = sum(physical_scores) / 5
            video_avg = session.get('video_score', physical_avg)
            final_avg = round((physical_avg + video_avg) / 2, 1)
            temp_data = session.get('temp_dog', {})
            
            new_record = DogRecord(
                owner_name=temp_data.get('owner_name', 'Unknown'),
                dog_name=temp_data.get('dog_name', 'Unknown'),
                dog_breed=temp_data.get('dog_breed', 'Unknown'),
                dog_age=temp_data.get('dog_age', 'Unknown'),
                average_score=final_avg, 
                stress_level=calculate_status(final_avg)
            )
            
            db.session.add(new_record)
            db.session.commit()
            
            session.pop('temp_dog', None)
            session.pop('video_score', None)
            
            flash("Assessment Complete! Record saved to MariaDB.")
            return redirect(url_for('records'))
            
        except (TypeError, ValueError):
            return render_template('questionnaire.html')

    return render_template('questionnaire.html')

@app.route('/records')
def records():
    if 'username' not in session: return redirect(url_for('login'))
    records = DogRecord.query.all()
    return render_template('records.html', rows=records)

@app.route('/delete_record/<int:id>', methods=['POST'])
def delete_record(id):
    record = DogRecord.query.get(id)
    if record:
        db.session.delete(record)
        db.session.commit()
        flash("Record deleted")
    return redirect(url_for('records'))

@app.route('/update_record', methods=['POST'])
def update_record():
    record_id = int(request.form.get('id'))
    record = DogRecord.query.get(record_id)
    if record:
        record.owner_name = request.form.get('owner_name')
        record.dog_name = request.form.get('dog_name')
        record.dog_breed = request.form.get('dog_breed')
        record.dog_age = request.form.get('dog_age')
        db.session.commit()
        flash("Record updated")
    return redirect(url_for('records'))

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'username' not in session: return redirect(url_for('login'))
    return render_template('settings.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/reset_assessment')
def reset_assessment():
    session.pop('temp_dog', None)
    session.pop('video_score', None)
    return redirect(url_for('monitoring'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)