#!/usr/bin/env python
"""
PlainSense Lite Server - Fast Startup
=====================================
This server starts instantly and loads ML models on first request.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import tempfile
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(app)

# Lazy-loaded API instances
_apis = {}

def get_api(domain: str = 'legal'):
    """Lazy load the PlainSense API for specific domain"""
    global _apis
    
    if domain not in _apis:
        print(f"\n🔄 First request for {domain.upper()} - loading API...")
        print("   (This may take 30-60 seconds on first load)")
        
        # Import here to avoid slow startup
        from api.plainsense_api import PlainSenseAPI
        _apis[domain] = PlainSenseAPI(domain=domain)
        print(f"✅ {domain.upper()} API loaded!\n")
    
    return _apis.get(domain)


@app.route('/')
def serve():
    """Serve React frontend"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def static_proxy(path):
    """Serve static files"""
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'api_version': '1.0',
        'loaded_apis': list(_apis.keys())
    })


@app.route('/api/process', methods=['POST'])
@app.route('/api/process/<domain>', methods=['POST'])
def process_document(domain=None):
    """Process uploaded document"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get domain from URL path or form data
        if domain is None:
            domain = request.form.get('domain', 'legal').lower()
        
        # Lazy load API (shows loading message on first request)
        api = get_api(domain)
        if api is None:
            return jsonify({'error': f'Unknown domain: {domain}'}), 400
        
        # Save file temporarily
        ext = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            print(f"[Processing] {file.filename} ({domain})")
            
            # Call the appropriate method based on domain
            if domain == 'legal':
                result = api.process_legal_document(tmp_path)
            elif domain == 'medical':
                result = api.process_medical_document(tmp_path)
            else:
                result = api.process_legal_document(tmp_path)  # default to legal
            
            print(f"[Done] {file.filename}")
            return jsonify(result)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/text', methods=['POST'])
def process_text():
    """Process text directly"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        domain = data.get('domain', 'legal').lower()
        api = get_api(domain)
        
        if api is None:
            return jsonify({'error': f'Unknown domain: {domain}'}), 400
        
        text = data['text']
        
        # Call appropriate method based on domain
        if domain == 'legal':
            result = api.process_legal_text(text)
        elif domain == 'medical':
            result = api.process_medical_text(text)
        else:
            result = api.process_legal_text(text)
        
        return jsonify(result)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("🚀 PlainSense Lite Server")
    print("=" * 50)
    print(f"📁 Static folder: {app.static_folder}")
    print(f"🌐 URL: http://localhost:5000")
    print("")
    print("⚡ Server starts instantly!")
    print("📝 ML models load on first document upload")
    print("=" * 50)
    print("")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
