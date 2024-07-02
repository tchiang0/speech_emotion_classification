from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return jsonify(error='No file part'), 400
    file = request.files['audio']
    if file.filename == '':
        return jsonify(error='No selected file'), 400
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        return jsonify(message='File uploaded successfully', filename=file.filename), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)
