from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from utils import classification_main_onnxruntime
import uuid
import os

def delete_old_images(upload_folder, num_to_delete):
    images = os.listdir(upload_folder)
    images.sort(key=lambda x: os.path.getmtime(os.path.join(upload_folder, x)))
    for i in range(num_to_delete):
        file_to_delete = os.path.join(upload_folder, images[i])
        try:
            os.remove(file_to_delete)
        except Exception as e:
            print(f'Error deleting file: {str(e)}')

def generate_secret_key():
    return os.urandom(24)

current_directory = os.path.dirname(os.path.abspath(__file__))
static_folder = os.path.join(current_directory, "static")

app = Flask(__name__,static_folder="static",template_folder="templates")
app.secret_key = generate_secret_key()
app.config['UPLOAD_FOLDER'] = os.path.join(current_directory, "static/uploads")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# if not os.path.exists(os.path.join(current_directory, "uploads")):
    # os.makedirs(os.path.join(current_directory, "uploads"))
if not os.path.exists(os.path.join(current_directory, "static", "uploads")):
    os.makedirs(os.path.join(current_directory, "static", "uploads"))


uploaded_images = os.listdir(app.config['UPLOAD_FOLDER'])
if len(uploaded_images) > 5:
    
    delete_old_images(app.config['UPLOAD_FOLDER'], len(uploaded_images) - 5)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())  # 使用随机生成的用户ID来区分不同用户

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # file_path = os.path.join(current_directory, "uploads", filename)
            image_path = os.path.join(current_directory, "static", "uploads", filename)
            file.save(image_path)
            # file.save(image_path)
            preds, output = classification_main_onnxruntime.cat_and_dog_classification(image_path)
            if preds[0] < 0.985:
                result = "The image is neither a dog nor a cat."
            elif output:
                preds_dog = classification_main_onnxruntime.dog_classification(image_path)
                result = format_results(preds_dog, "Dog")
            elif output == 0:
                preds_cat = classification_main_onnxruntime.cat_classification(image_path)
                result = format_results(preds_cat, "Cat")
            image_path = os.path.join("uploads", filename)
            image_path = image_path.replace("\\", "/")
            session['result'] = result
            session['image_path'] = image_path
            return redirect(url_for('results'))  # Redirect to 'results' page after processing
        else:
            flash('Invalid file type', 'error')
            return redirect(request.url)
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html', result=session.get('result', None), image_path=session.get('image_path', None))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/clear', methods=['GET', 'POST'])
def clear():
    if request.method == 'POST':
        session.pop('result', None)
        session.pop('image_path', None)
        return redirect('/')

    return "Page cleared successfully"

def format_results(predictions, category):
    result_text = f"{category} (Top 5 Results):<br>"
    for i, (class_name, prob) in enumerate(predictions[:5], start=1):
        result_text += f"{i}. {class_name}: {prob * 100:.2f}%<br>"
    return result_text

if __name__ == '__main__':
    app.run(debug=True)

    