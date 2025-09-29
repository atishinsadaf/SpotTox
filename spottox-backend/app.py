# app.py - SpotTox Backend Server
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests from React frontend

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'json'}

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created upload folder: {UPLOAD_FOLDER}")

def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_thread_file(file_path):
    """
    Read and parse thread file content.
    Handles both JSON and plain text files.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Try to parse as JSON first
        try:
            thread_data = json.loads(content)
            file_type = 'json'
            print(f"Successfully parsed JSON file")
        except json.JSONDecodeError:
            # If not JSON, treat as plain text
            thread_data = content.split('\n')
            file_type = 'txt'
            print(f"Parsed as text file with {len(thread_data)} lines")
            
        return {
            'success': True,
            'file_type': file_type,
            'content': thread_data,
            'message_count': len(thread_data) if isinstance(thread_data, list) else 1,
            'timestamp': datetime.now().isoformat(),
            'file_size_bytes': os.path.getsize(file_path)
        }
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@app.route('/', methods=['GET'])
def home():
    """API Home page - shows available endpoints."""
    return jsonify({
        'message': 'SpotTox Backend API - Thread Toxicity Detection System',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'GET /': 'API information (this page)',
            'GET /health': 'Health check',
            'POST /upload': 'Upload thread file for analysis',
            'GET /threads': 'List all uploaded threads',
            'POST /analyze': 'Analyze thread for toxicity (placeholder)'
        },
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'upload_folder': UPLOAD_FOLDER
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'SpotTox Backend'
    })

@app.route('/upload', methods=['POST'])
def upload_thread():
    """Handle thread file uploads from React frontend."""
    print("Upload request received")
    
    # Check if file was included in request
    if 'file' not in request.files:
        print("No file in request")
        return jsonify({'error': 'No file provided in request'}), 400
    
    file = request.files['file']
    print(f"File received: {file.filename}")
    
    # Validate file selection
    if file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file type
    if file and allowed_file(file.filename):
        try:
            # Save file to uploads directory
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            print(f"File saved to: {file_path}")
            
            # Read and parse the uploaded file
            result = read_thread_file(file_path)
            
            if result['success']:
                print(f"Successfully processed {filename}")
                return jsonify({
                    'message': f'File "{filename}" uploaded and processed successfully',
                    'filename': filename,
                    'file_info': result,
                    'next_steps': 'Ready for toxicity analysis (not implemented yet)'
                }), 200
            else:
                print(f"Failed to process {filename}: {result['error']}")
                return jsonify({
                    'error': f'Failed to process file: {result["error"]}',
                    'filename': filename
                }), 500
                
        except Exception as e:
            print(f"Upload error: {str(e)}")
            return jsonify({
                'error': f'Upload failed: {str(e)}',
                'filename': file.filename
            }), 500
    else:
        print(f"Invalid file type: {file.filename}")
        return jsonify({
            'error': 'Invalid file type',
            'allowed_types': list(ALLOWED_EXTENSIONS),
            'received_filename': file.filename
        }), 400

@app.route('/threads', methods=['GET'])
def list_threads():
    """List all uploaded thread files with metadata."""
    try:
        files = []
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                if allowed_file(filename):
                    file_path = os.path.join(UPLOAD_FOLDER, filename)
                    file_stats = os.stat(file_path)
                    files.append({
                        'filename': filename,
                        'size_bytes': file_stats.st_size,
                        'size_kb': round(file_stats.st_size / 1024, 2),
                        'uploaded': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                        'file_type': filename.split('.')[-1].lower()
                    })
        
        return jsonify({
            'threads': sorted(files, key=lambda x: x['uploaded'], reverse=True),
            'total_count': len(files),
            'upload_folder': UPLOAD_FOLDER,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error listing threads: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_thread():
    """Placeholder endpoint for future toxicity analysis."""
    try:
        data = request.get_json()
        filename = data.get('filename', 'unknown') if data else 'unknown'
        
        print(f"Analysis requested for: {filename}")
        
        # Mock response for demonstration
        return jsonify({
            'message': f'Analysis initiated for "{filename}"',
            'status': 'processing',
            'analysis_type': 'toxicity_detection',
            'estimated_time': '30-60 seconds',
            'note': 'This is a placeholder response. ML processing will be implemented in future milestones.',
            'timestamp': datetime.now().isoformat(),
            'mock_results': {
                'toxicity_score': 0.23,
                'confidence': 0.85,
                'flagged_messages': 1
            }
        })
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*50)
    print("🚀 Starting SpotTox Backend Server")
    print("="*50)
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📄 Allowed file types: {ALLOWED_EXTENSIONS}")
    print(f"🌐 Server will run on: http://localhost:5001")
    print(f"🔍 Health check: http://localhost:5001/health")
    print("="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5001)