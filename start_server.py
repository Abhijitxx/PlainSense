#!/usr/bin/env python
"""
PlainSense Server Launcher
Handles signals properly to prevent interruption during ML model loading.
"""
import sys
import os

# ============================================================
# IMPORTANT: Set signal handling BEFORE any other imports
# This prevents Ctrl+C during module loading
# ============================================================
import signal
if os.name == 'nt':
    # On Windows, completely disable SIGINT during startup
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

# Ensure proper path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=" * 60)
print("🚀 PlainSense Server Launcher")
print("=" * 60)
print("\n⏳ Loading server components...")
print("   (This may take 30-60 seconds due to ML models)")
print("   ⚠️  Please wait - ignoring Ctrl+C during loading\n")

try:
    # Now do the imports with signals disabled
    print("   📦 Loading Flask...")
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS
    import tempfile
    import traceback
    from werkzeug.utils import secure_filename
    print("   ✓ Flask loaded")
    
    print("   📦 Loading PlainSense API (heavy ML models)...")
    from api.plainsense_api import PlainSenseAPI
    print("   ✓ PlainSense API loaded")
    
    # Create app
    app = Flask(__name__, static_folder='frontend/build', static_url_path='')
    CORS(app)
    
    # Import all routes from server
    print("   📦 Loading server routes...")
    # We'll copy the routes here inline
    
except Exception as e:
    print(f"\n❌ Error during import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# Restore signal handlers now that imports are done
# ============================================================
if os.name == 'nt':
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)

print("\n✅ All components loaded successfully!")
print("=" * 60)

# Copy essential routes from server.py
_legal_api = None
_medical_api = None

def get_api(domain: str = 'legal'):
    """Lazy load the PlainSense API for specific domain"""
    global _legal_api, _medical_api
    
    if domain == 'legal':
        if _legal_api is None:
            print("🔄 Initializing PlainSense API for LEGAL documents...")
            _legal_api = PlainSenseAPI(domain='legal')
            print("✅ Legal API initialized!")
        return _legal_api
    elif domain == 'medical':
        if _medical_api is None:
            print("🔄 Initializing PlainSense API for MEDICAL documents...")
            _medical_api = PlainSenseAPI(domain='medical')
            print("✅ Medical API initialized!")
        return _medical_api
    return None

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
        'legal_api': _legal_api is not None,
        'medical_api': _medical_api is not None
    })

@app.route('/api/process', methods=['POST'])
def process_document():
    """Process uploaded document"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get domain from request (default: legal)
        domain = request.form.get('domain', 'legal').lower()
        
        # Get API for domain
        api = get_api(domain)
        if api is None:
            return jsonify({'error': f'Unknown domain: {domain}'}), 400
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            print(f"\n[DEBUG] Processing file: {file.filename}")
            print(f"[DEBUG] Domain: {domain}")
            print(f"[DEBUG] Temp path: {tmp_path}")
            
            # Process document
            result = api.process_document(tmp_path)
            
            print(f"[DEBUG] Processing complete!")
            
            return jsonify(result)
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/simplify', methods=['POST'])
def simplify_text():
    """Simplify text directly"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        domain = data.get('domain', 'legal').lower()
        api = get_api(domain)
        
        if api is None:
            return jsonify({'error': f'Unknown domain: {domain}'}), 400
        
        text = data['text']
        simplified = api.simplifier.simplify_text(text)
        
        return jsonify({
            'original': text,
            'simplified': simplified
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n🚀 Starting Flask server on http://localhost:5000")
    print("📝 Press Ctrl+C to stop\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n⛔ Server stopped by user")
        sys.exit(0)
