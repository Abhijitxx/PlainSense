# PlainSense - Document Simplification System

**Final Year Project** - AI-powered system for simplifying legal and medical documents

## 📋 Overview

PlainSense is an intelligent document processing system that:
- Extracts text from documents (PDF, images, DOCX)
- Identifies document type (Legal vs Medical)
- Segments documents into clauses/sections
- Simplifies complex legal/medical language to plain English
- Translates to Hindi and Tamil (formal and colloquial)
- Detects risk levels in clauses
- Provides medical term explanations

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PlainSense API                               │
│                   (plainsense_api.py)                           │
├─────────────────┬──────────────────────┬────────────────────────┤
│   OCR Pipeline  │    LLM Simplifier    │   Medical Dictionary   │
│   (Tesseract)   │     (FLAN-T5)        │   (medical_dict.py)    │
├─────────────────┼──────────────────────┼────────────────────────┤
│ Clause Segmenter│    Risk Detector     │  Medical Parser        │
│ (clause_seg.py) │    (LegalBERT)       │  (med_parser.py)       │
├─────────────────┴──────────────────────┴────────────────────────┤
│                    Translation Models                            │
│              (MarianMT: Hindi & Tamil)                          │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Key Features

### Legal Documents
- ✅ Clause segmentation
- ✅ Plain English simplification
- ✅ Colloquial/friendly simplification
- ✅ Risk detection (CRITICAL/HIGH/MEDIUM/LOW)
- ✅ Key term extraction
- ✅ Entity preservation checking

### Medical Documents
- ✅ Lab report parsing
- ✅ Medical term simplification
- ✅ Risk assessment for abnormal values
- ✅ Medical dictionary integration

### Multi-Language Support
- ✅ Hindi formal translation
- ✅ Hindi colloquial translation
- ✅ Tamil formal translation
- ✅ Tamil colloquial translation

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/plainsense.git
cd plainsense

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Tesseract OCR (optional, for scanned documents)

## 🎯 Quick Start

### Using the API

```python
from plainsense_api import PlainSenseAPI

# Initialize
api = PlainSenseAPI()

# Process legal document
result = api.process_legal_document("Your rental agreement text here...")
print(result.clauses)  # Simplified clauses
print(result.summary)  # Risk summary

# Process medical document
result = api.process_medical_document("Hemoglobin: 8.5 g/dL (Normal: 12-17)")
print(result.clauses)  # Simplified medical results
print(result.medical_explanations)  # Dictionary explanations
```

### Running the Demo

```bash
python demo.py
```

### Running Tests

```bash
python test_full_system.py
```

## 📁 Project Structure

```
PlainSense/
├── plainsense_api.py      # Unified API for the system
├── llm_simplifier.py      # Core LLM-based simplification
├── medical_dictionary.py  # Medical term explanations
├── clause_segmenter.py    # Document segmentation
├── medical_report_parser.py # Lab report parsing
├── ocr_pipeline.py        # OCR text extraction
├── pipeline.py            # Master integration pipeline
├── demo.py                # Interactive demonstration
├── test_full_system.py    # Comprehensive tests
├── streamlit_app.py       # Web interface
├── frontend/              # React frontend
└── requirements.txt       # Python dependencies
```

## 🔧 Configuration

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Simplification | FLAN-T5-base | Text simplification |
| Legal Risk | LegalBERT | Risk embedding |
| Medical NER | BioBERT | Named entity recognition |
| Hindi Translation | MarianMT en-hi | English to Hindi |
| Tamil Translation | MarianMT en-ta | English to Tamil |

### Performance

- Legal clause processing: ~5-15s per clause (CPU)
- Medical report processing: ~3-10s per section (CPU)
- Translation: ~2-5s per text (first load ~2min)

## 📊 Output Format

### ClauseResult Structure

```json
{
  "original": "The original clause text...",
  "english": {
    "plain": "Plain English version...",
    "colloquial": "Friendly version..."
  },
  "hindi": {
    "formal": "हिंदी औपचारिक...",
    "colloquial": "हिंदी अनौपचारिक..."
  },
  "tamil": {
    "formal": "தமிழ் முறையான...",
    "colloquial": "தமிழ் பேச்சு..."
  },
  "risk": {
    "level": "HIGH",
    "score": 0.75,
    "explanation": "Short notice period of 7 days"
  },
  "key_terms": ["Rs. 50,000", "30 days", "penalty"],
  "entities_preserved": true,
  "preservation_warnings": []
}
```

## 🔬 Medical Dictionary

The system includes a comprehensive medical dictionary:

```python
# Explain a medical term
api.explain_medical_term("hemoglobin")
# Returns: {
#   "simple_name": "Oxygen carrier in blood",
#   "description": "A protein in red blood cells...",
#   "normal_range": "12-17 g/dL"
# }

# Interpret a lab result
api.interpret_lab_result("Hemoglobin", 8.5, "g/dL")
# Returns: {
#   "status": "LOW",
#   "meaning": "Indicates anemia..."
# }
```

## 🧪 Risk Detection

### Legal Risk Levels

| Level | Description |
|-------|-------------|
| CRITICAL | Complete rights waiver, immediate eviction |
| HIGH | Short notice (7-14 days), heavy penalties |
| MEDIUM | Standard penalties, normal fees |
| LOW | Fair terms, tenant protections |
| NONE | Balanced clause |

### Medical Risk Levels

| Level | Description |
|-------|-------------|
| CRITICAL | Values 3x+ outside normal, life-threatening |
| HIGH | Significantly abnormal values |
| MEDIUM | Slightly outside normal range |
| LOW | Borderline values |
| NONE | All values normal |

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 👥 Authors

- PlainSense Team - Final Year Project

## 🙏 Acknowledgments

- Hugging Face for Transformers library
- Google for FLAN-T5 model
- Helsinki NLP for MarianMT translation models
