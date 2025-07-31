import os
from flask import Flask, request, render_template, jsonify
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF library
from werkzeug.utils import secure_filename

# --- App Configuration ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Create the uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the Sentence Transformer model once when the app starts
# This is crucial for performance, so it doesn't reload on every request.
# 'all-MiniLM-L6-v2' is a good, lightweight choice.
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Helper Functions ---
def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    return text

# --- Flask Routes ---
@app.route('/')
def index():
    """Render the main page with the upload form."""
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match_cvs():
    """Handle the file uploads and perform the matching."""
    job_description = request.form.get('job_description')
    if not job_description:
        return jsonify({'error': 'Job description is required.'}), 400

    # Handle multiple file uploads
    cv_files = request.files.getlist('cv_files')
    if not cv_files or all(f.filename == '' for f in cv_files):
        return jsonify({'error': 'Please upload one or more CV files.'}), 400

    job_embedding = model.encode(job_description, convert_to_tensor=True)
    cv_data = []

    for file in cv_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Extract text from the uploaded PDF
            cv_text = extract_text_from_pdf(file_path)
            if cv_text:
                cv_embedding = model.encode(cv_text, convert_to_tensor=True)
                
                # Calculate the similarity score
                score = util.cos_sim(job_embedding, cv_embedding).item()

                cv_data.append({
                    'filename': filename,
                    'score': f"{score * 100:.2f}%",  # Format as a percentage
                    'raw_score': score  # Keep the raw score for sorting
                })

            # Clean up the temporary file
            os.remove(file_path)
        
        else:
            # Handle invalid file types
            cv_data.append({
                'filename': file.filename,
                'error': 'Invalid file type. Only PDF files are allowed.'
            })

    # Sort results by score in descending order
    cv_data.sort(key=lambda x: x.get('raw_score', -1), reverse=True)

    return jsonify({'results': cv_data})

if __name__ == '__main__':
    app.run(debug=True)