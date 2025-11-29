"""
PlainSense Flask Backend Server
Serves the React frontend and provides API endpoints for document processing
"""

import os
import sys
import tempfile
import traceback

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.plainsense_api import PlainSenseAPI

# Fix static folder path for new directory structure
app = Flask(__name__, static_folder='../frontend/build', static_url_path='')
CORS(app)  # Enable CORS for development

# Initialize PlainSense API instances per domain (lazy loading)
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
    else:
        # Default to legal
        return get_api('legal')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'txt', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============== API Routes ==============

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'service': 'PlainSense API'
    })


@app.route('/api/validate/domain', methods=['POST'])
def validate_domain():
    """Validate document domain (legal vs medical)"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        expected_domain = request.form.get('expected_domain', 'legal')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            api = get_api('legal')  # Use legal API for domain detection
            # Extract text from file
            text = api.extract_text(tmp_path)
            
            # Detect domain
            detected_domain = api.classify_domain(text)
            
            is_valid = detected_domain.lower() == expected_domain.lower()
            
            return jsonify({
                'valid': is_valid,
                'detected_domain': detected_domain,
                'expected_domain': expected_domain,
                'error': None if is_valid else f'Document appears to be {detected_domain}, not {expected_domain}'
            })
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/legal', methods=['POST'])
def process_legal():
    """Process a legal document"""
    try:
        api = get_api('legal')
        
        # Handle file upload
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'}), 400
            
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name
            
            print(f"[DEBUG] Processing file: {tmp_path}")
            
            try:
                # Process document WITHOUT translations (faster initial load)
                result = api.process_legal_document(tmp_path, include_translations=False)
                print(f"[DEBUG] Result clauses count: {len(result.clauses)}")
                print(f"[DEBUG] Result errors: {result.errors}")
            finally:
                os.unlink(tmp_path)
        else:
            # Handle text input - check form data first, then JSON
            text = request.form.get('text', '')
            if not text:
                data = request.get_json(silent=True) or {}
                text = data.get('text', '')
            
            if not text:
                return jsonify({'error': 'No file or text provided'}), 400
            
            print(f"[DEBUG] Processing text of length: {len(text)}")
            print(f"[DEBUG] Text preview: {text[:200]}...")
            
            # Process text WITHOUT translations (faster initial load)
            result = api.process_legal_document(text, include_translations=False)
            print(f"[DEBUG] Result clauses count: {len(result.clauses)}")
            print(f"[DEBUG] Result errors: {result.errors}")
        
        # Check for errors
        if result.errors:
            # Check if it might be a medical document based on errors
            error_text = ' '.join(result.errors).lower()
            if 'medical' in error_text:
                return jsonify({
                    'success': False,
                    'error': 'This appears to be a medical document. Please use the Medical tab.',
                }), 400
        
        # Format response
        clauses = []
        for clause in result.clauses:
            clause_dict = clause if isinstance(clause, dict) else clause
            
            # Extract risk info
            risk_level = 'low'
            risk_explanation = ''
            if isinstance(clause_dict, dict):
                risk_info = clause_dict.get('risk', {})
                if isinstance(risk_info, dict):
                    risk_level = risk_info.get('level', 'low')
                    risk_explanation = risk_info.get('explanation', '')
            
            # Extract text - handle both formats
            original = clause_dict.get('original', '')
            english_data = clause_dict.get('english', {})
            simplified = english_data.get('plain', '') if isinstance(english_data, dict) else clause_dict.get('simplified', '')
            
            # Extract translations
            hindi_data = clause_dict.get('hindi', {})
            tamil_data = clause_dict.get('tamil', {})
            
            clauses.append({
                'original': original,
                'simplified': simplified,
                'english': {
                    'plain': simplified
                },
                'hindi': hindi_data if isinstance(hindi_data, dict) else {'formal': '', 'colloquial': ''},
                'tamil': tamil_data if isinstance(tamil_data, dict) else {'formal': '', 'colloquial': ''},
                'risk': {
                    'level': risk_level,
                    'explanation': risk_explanation
                },
                'key_terms': clause_dict.get('key_terms', [])
            })
        
        return jsonify({
            'success': True,
            'document_type': 'legal',
            'domain': 'legal',
            'clauses': clauses,
            'law_references': result.law_references if hasattr(result, 'law_references') else [],
            'summary': result.summary if hasattr(result, 'summary') else {
                'total_clauses': len(clauses),
                'high_risk_count': sum(1 for c in clauses if c['risk']['level'] in ['high', 'critical']),
                'medium_risk_count': sum(1 for c in clauses if c['risk']['level'] == 'medium'),
                'low_risk_count': sum(1 for c in clauses if c['risk']['level'] == 'low'),
                'overall_risk': 'HIGH' if any(c['risk']['level'] in ['high', 'critical'] for c in clauses) else 'LOW'
            }
        })
            
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/process/medical', methods=['POST'])
def process_medical():
    """Process a medical document"""
    try:
        api = get_api('medical')
        
        # Handle file upload
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'}), 400
            
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                # Process document using API
                result = api.process_medical_document(tmp_path, include_translations=True)
            finally:
                os.unlink(tmp_path)
        else:
            # Handle text input from JSON body
            data = request.get_json(silent=True) or {}
            text = data.get('text', '')
            
            if not text:
                return jsonify({'error': 'No file or text provided'}), 400
            
            # Process text using API
            result = api.process_medical_document(text, include_translations=True)
        
        # Check for errors
        if result.errors:
            # Check if it might be a legal document
            error_text = ' '.join(result.errors).lower()
            if 'legal' in error_text:
                return jsonify({
                    'success': False,
                    'error': 'This appears to be a legal document. Please use the Legal tab.',
                }), 400
        
        # Format response
        results = []
        abnormal_count = 0
        critical_count = 0
        
        for item in result.clauses:
            item_dict = item if isinstance(item, dict) else item
            
            risk_level = 'normal'
            risk_explanation = ''
            if isinstance(item_dict, dict):
                risk_info = item_dict.get('risk', {})
                if isinstance(risk_info, dict):
                    risk_level = risk_info.get('level', 'normal')
                    risk_explanation = risk_info.get('explanation', '')
                elif hasattr(item_dict, 'risk_level'):
                    risk_level = str(item_dict.get('risk_level', 'normal')).lower()
            
            if risk_level in ['high', 'critical']:
                critical_count += 1
                abnormal_count += 1
            elif risk_level == 'medium':
                abnormal_count += 1
            
            results.append({
                'original': item_dict.get('original', ''),
                'simplified': item_dict.get('simplified', item_dict.get('english', {}).get('plain', '')),
                'english': {
                    'plain': item_dict.get('simplified', item_dict.get('english', {}).get('plain', ''))
                },
                'hindi': item_dict.get('hindi', item_dict.get('translations', {}).get('hindi', {})),
                'tamil': item_dict.get('tamil', item_dict.get('translations', {}).get('tamil', {})),
                'risk': {
                    'level': risk_level,
                    'explanation': risk_explanation
                },
                'test_name': item_dict.get('test_name', ''),
                'value': item_dict.get('value', ''),
                'normal_range': item_dict.get('normal_range', ''),
            })
        
        return jsonify({
            'success': True,
            'document_type': 'medical',
            'domain': 'medical',
            'results': results,
            'summary': {
                'total_tests': len(results),
                'abnormal_count': abnormal_count,
                'critical_count': critical_count
            }
        })
            
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/process/text', methods=['POST'])
def process_text():
    """Process text directly without file upload"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        text = data.get('text', '')
        domain = data.get('domain', 'legal')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        api = get_api(domain)  # Use the specified domain
        
        # Process based on domain
        if domain == 'legal':
            result = api.process_legal_document(text, include_translations=True)
            
            clauses = []
            for clause in result.clauses:
                clause_dict = clause if isinstance(clause, dict) else clause
                risk_level = 'low'
                risk_explanation = ''
                if isinstance(clause_dict, dict):
                    risk_info = clause_dict.get('risk', {})
                    if isinstance(risk_info, dict):
                        risk_level = risk_info.get('level', 'low')
                        risk_explanation = risk_info.get('explanation', '')
                
                clauses.append({
                    'original': clause_dict.get('original', ''),
                    'simplified': clause_dict.get('simplified', clause_dict.get('english', {}).get('plain', '')),
                    'english': {'plain': clause_dict.get('simplified', clause_dict.get('english', {}).get('plain', ''))},
                    'hindi': clause_dict.get('hindi', clause_dict.get('translations', {}).get('hindi', {})),
                    'tamil': clause_dict.get('tamil', clause_dict.get('translations', {}).get('tamil', {})),
                    'risk': {'level': risk_level, 'explanation': risk_explanation},
                    'key_terms': clause_dict.get('key_terms', [])
                })
            
            return jsonify({
                'success': True,
                'document_type': 'legal',
                'domain': domain,
                'clauses': clauses,
                'law_references': result.law_references if hasattr(result, 'law_references') else [],
                'summary': result.summary if hasattr(result, 'summary') else {'total_clauses': len(clauses)}
            })
        else:
            result = api.process_medical_document(text, include_translations=True)
            
            results = []
            for item in result.clauses:
                item_dict = item if isinstance(item, dict) else item
                risk_level = 'normal'
                risk_explanation = ''
                if isinstance(item_dict, dict):
                    risk_info = item_dict.get('risk', {})
                    if isinstance(risk_info, dict):
                        risk_level = risk_info.get('level', 'normal')
                        risk_explanation = risk_info.get('explanation', '')
                
                results.append({
                    'original': item_dict.get('original', ''),
                    'simplified': item_dict.get('simplified', item_dict.get('english', {}).get('plain', '')),
                    'english': {'plain': item_dict.get('simplified', item_dict.get('english', {}).get('plain', ''))},
                    'hindi': item_dict.get('hindi', item_dict.get('translations', {}).get('hindi', {})),
                    'tamil': item_dict.get('tamil', item_dict.get('translations', {}).get('tamil', {})),
                    'risk': {'level': risk_level, 'explanation': risk_explanation}
                })
            
            return jsonify({
                'success': True,
                'document_type': 'medical',
                'domain': domain,
                'results': results,
                'summary': result.summary if hasattr(result, 'summary') else {'total_tests': len(results)}
            })
            
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/translate', methods=['POST'])
def translate():
    """Translate text to Hindi or Tamil - loads model on demand"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        text = data.get('text', '')
        target_lang = data.get('target_lang', 'hi')  # 'hi' for Hindi, 'ta' for Tamil
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        api = get_api('legal')  # Translation models are shared
        
        # Map language names to codes
        lang_map = {
            'hindi': 'hi',
            'tamil': 'ta',
            'hi': 'hi',
            'ta': 'ta'
        }
        lang_code = lang_map.get(target_lang.lower(), 'hi')
        
        print(f"[DEBUG] Translating to {lang_code}: {text[:100]}...")
        
        # Use the simplifier's translate method (loads model on demand)
        translation = api.simplifier._translate(text, lang_code)
        
        return jsonify({
            'success': True,
            'translation': translation,
            'source': text,
            'target_lang': lang_code
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============== Serve React App ==============

@app.route('/')
def serve():
    """Serve React app"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


# ============== Error Handlers ==============

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ============== Main ==============

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                     🧠 PlainSense Server                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Starting server on http://localhost:{port}                        ║
║  API endpoints available at http://localhost:{port}/api            ║
║                                                                  ║
║  Endpoints:                                                      ║
║    POST /api/process/legal   - Process legal documents           ║
║    POST /api/process/medical - Process medical documents         ║
║    POST /api/process/text    - Process raw text                  ║
║    POST /api/validate/domain - Validate document domain          ║
║    POST /api/translate       - Translate text                    ║
║    GET  /api/health          - Health check                      ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    app.run(host='0.0.0.0', port=port, debug=debug)
