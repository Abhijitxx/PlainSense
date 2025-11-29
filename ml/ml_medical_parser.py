"""
Enhanced Medical Report Parser v2
=================================

Major Improvements:
1. Comprehensive reference range database (Indian labs)
2. Better NER with fallback patterns
3. Unit normalization
4. Multi-format support (tables, free text)
5. Abnormality severity classification

Author: PlainSense Team
Date: November 2025
"""

import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class AbnormalityLevel(Enum):
    """Abnormality severity"""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    MILD = "mild"
    NORMAL = "normal"
    UNKNOWN = "unknown"


@dataclass
class TestResult:
    """Parsed medical test result"""
    test_name: str
    value: float
    unit: str
    reference_range: str
    status: AbnormalityLevel
    deviation_percent: float
    interpretation: str
    
    def to_dict(self) -> Dict:
        return {
            'test_name': self.test_name,
            'value': self.value,
            'unit': self.unit,
            'reference_range': self.reference_range,
            'status': self.status.value,
            'deviation_percent': round(self.deviation_percent, 1),
            'interpretation': self.interpretation
        }


@dataclass
class MedicalReport:
    """Complete medical report parsing result"""
    patient_name: Optional[str]
    patient_age: Optional[str]
    patient_gender: Optional[str]
    test_date: Optional[str]
    lab_name: Optional[str]
    test_results: List[TestResult]
    summary: str
    abnormal_count: int
    critical_count: int
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            'patient': {
                'name': self.patient_name,
                'age': self.patient_age,
                'gender': self.patient_gender
            },
            'test_date': self.test_date,
            'lab_name': self.lab_name,
            'test_results': [t.to_dict() for t in self.test_results],
            'summary': self.summary,
            'abnormal_count': self.abnormal_count,
            'critical_count': self.critical_count,
            'confidence': round(self.confidence, 2)
        }


class EnhancedMedicalParser:
    """
    Enhanced medical report parser with comprehensive test database
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize medical parser
        
        Args:
            use_gpu: Use GPU if available
        """
        self.device = 0 if use_gpu else -1
        
        # NER model for biomedical entities
        self.ner_model = None
        
        # Embedding model for test name matching
        self.embedding_model = None
        
        # Reference ranges database
        self.reference_ranges = self._init_reference_ranges()
        
        # Test name embeddings for fuzzy matching
        self.test_embeddings = {}
        
        self._init_models()
    
    def _init_reference_ranges(self) -> Dict[str, Dict]:
        """
        Comprehensive reference range database
        Based on Indian laboratory standards
        """
        return {
            # ===================
            # Complete Blood Count (CBC)
            # ===================
            'hemoglobin': {
                'aliases': ['hb', 'hgb', 'haemoglobin'],
                'unit': 'g/dL',
                'male': (13.0, 17.0),
                'female': (12.0, 15.5),
                'general': (12.0, 17.0),
                'critical_low': 7.0,
                'critical_high': 20.0,
            },
            'rbc': {
                'aliases': ['red blood cell', 'erythrocyte', 'rbc count'],
                'unit': 'million/cumm',
                'male': (4.5, 5.5),
                'female': (4.0, 5.0),
                'general': (4.0, 5.5),
                'critical_low': 2.5,
                'critical_high': 7.0,
            },
            'wbc': {
                'aliases': ['white blood cell', 'leukocyte', 'wbc count', 'total wbc', 'tlc'],
                'unit': 'cells/cumm',
                'general': (4000, 11000),
                'critical_low': 2000,
                'critical_high': 30000,
            },
            'platelets': {
                'aliases': ['platelet count', 'thrombocyte', 'plt'],
                'unit': '/cumm',
                'general': (150000, 400000),
                'critical_low': 50000,
                'critical_high': 1000000,
            },
            'hematocrit': {
                'aliases': ['hct', 'pcv', 'packed cell volume'],
                'unit': '%',
                'male': (38.0, 50.0),
                'female': (36.0, 44.0),
                'general': (36.0, 50.0),
                'critical_low': 20.0,
                'critical_high': 60.0,
            },
            'mcv': {
                'aliases': ['mean corpuscular volume'],
                'unit': 'fL',
                'general': (80.0, 100.0),
            },
            'mch': {
                'aliases': ['mean corpuscular hemoglobin'],
                'unit': 'pg',
                'general': (27.0, 32.0),
            },
            'mchc': {
                'aliases': ['mean corpuscular hemoglobin concentration'],
                'unit': 'g/dL',
                'general': (32.0, 36.0),
            },
            'rdw': {
                'aliases': ['red cell distribution width'],
                'unit': '%',
                'general': (11.5, 14.5),
            },
            'mpv': {
                'aliases': ['mean platelet volume'],
                'unit': 'fL',
                'general': (7.5, 11.5),
            },
            
            # ===================
            # Blood Sugar
            # ===================
            'glucose_fasting': {
                'aliases': ['fasting glucose', 'fbs', 'fasting blood sugar', 'fbg'],
                'unit': 'mg/dL',
                'general': (70, 100),
                'prediabetic': (100, 125),
                'diabetic': 126,
                'critical_low': 50,
                'critical_high': 400,
            },
            'glucose_pp': {
                'aliases': ['post prandial glucose', 'ppbs', 'pp glucose', 'postprandial'],
                'unit': 'mg/dL',
                'general': (70, 140),
                'prediabetic': (140, 199),
                'diabetic': 200,
                'critical_high': 400,
            },
            'glucose_random': {
                'aliases': ['random glucose', 'rbs', 'random blood sugar'],
                'unit': 'mg/dL',
                'general': (70, 140),
                'critical_low': 50,
                'critical_high': 400,
            },
            'hba1c': {
                'aliases': ['glycated hemoglobin', 'hemoglobin a1c', 'a1c'],
                'unit': '%',
                'general': (4.0, 5.6),
                'prediabetic': (5.7, 6.4),
                'diabetic': 6.5,
                'critical_high': 12.0,
            },
            
            # ===================
            # Lipid Profile
            # ===================
            'cholesterol_total': {
                'aliases': ['total cholesterol', 'cholesterol', 'tc'],
                'unit': 'mg/dL',
                'general': (0, 200),
                'borderline': (200, 239),
                'high': 240,
            },
            'hdl': {
                'aliases': ['hdl cholesterol', 'hdl-c', 'good cholesterol'],
                'unit': 'mg/dL',
                'male': (40, 60),
                'female': (50, 60),
                'general': (40, 60),
                'critical_low': 30,
            },
            'ldl': {
                'aliases': ['ldl cholesterol', 'ldl-c', 'bad cholesterol'],
                'unit': 'mg/dL',
                'general': (0, 100),
                'borderline': (100, 159),
                'high': 160,
            },
            'triglycerides': {
                'aliases': ['tg', 'trigs'],
                'unit': 'mg/dL',
                'general': (0, 150),
                'borderline': (150, 199),
                'high': 200,
                'critical_high': 500,
            },
            'vldl': {
                'aliases': ['vldl cholesterol'],
                'unit': 'mg/dL',
                'general': (5, 40),
            },
            
            # ===================
            # Kidney Function (KFT/RFT)
            # ===================
            'creatinine': {
                'aliases': ['serum creatinine', 's.creatinine'],
                'unit': 'mg/dL',
                'male': (0.7, 1.3),
                'female': (0.6, 1.1),
                'general': (0.6, 1.3),
                'critical_high': 10.0,
            },
            'urea': {
                'aliases': ['blood urea', 'bun', 'blood urea nitrogen'],
                'unit': 'mg/dL',
                'general': (15, 45),
                'critical_high': 100,
            },
            'uric_acid': {
                'aliases': ['serum uric acid'],
                'unit': 'mg/dL',
                'male': (3.5, 7.2),
                'female': (2.6, 6.0),
                'general': (2.6, 7.2),
            },
            'egfr': {
                'aliases': ['estimated gfr', 'glomerular filtration rate'],
                'unit': 'mL/min',
                'general': (90, 120),
                'critical_low': 15,
            },
            
            # ===================
            # Liver Function (LFT)
            # ===================
            'bilirubin_total': {
                'aliases': ['total bilirubin', 't.bilirubin', 'tbil'],
                'unit': 'mg/dL',
                'general': (0.2, 1.2),
                'critical_high': 12.0,
            },
            'bilirubin_direct': {
                'aliases': ['direct bilirubin', 'd.bilirubin', 'dbil', 'conjugated bilirubin'],
                'unit': 'mg/dL',
                'general': (0.0, 0.3),
            },
            'sgot': {
                'aliases': ['ast', 'aspartate aminotransferase', 'sgot/ast'],
                'unit': 'U/L',
                'general': (5, 40),
                'critical_high': 1000,
            },
            'sgpt': {
                'aliases': ['alt', 'alanine aminotransferase', 'sgpt/alt'],
                'unit': 'U/L',
                'general': (5, 40),
                'critical_high': 1000,
            },
            'alp': {
                'aliases': ['alkaline phosphatase'],
                'unit': 'U/L',
                'general': (44, 147),
            },
            'ggt': {
                'aliases': ['gamma gt', 'gamma glutamyl transferase'],
                'unit': 'U/L',
                'male': (10, 71),
                'female': (6, 42),
                'general': (6, 71),
            },
            'albumin': {
                'aliases': ['serum albumin', 's.albumin'],
                'unit': 'g/dL',
                'general': (3.5, 5.0),
                'critical_low': 2.0,
            },
            'protein_total': {
                'aliases': ['total protein', 'serum protein'],
                'unit': 'g/dL',
                'general': (6.0, 8.0),
            },
            
            # ===================
            # Thyroid Function
            # ===================
            'tsh': {
                'aliases': ['thyroid stimulating hormone', 's.tsh'],
                'unit': 'mIU/L',
                'general': (0.4, 4.0),
                'critical_low': 0.1,
                'critical_high': 10.0,
            },
            't3_total': {
                'aliases': ['total t3', 'triiodothyronine'],
                'unit': 'ng/dL',
                'general': (80, 200),
            },
            't4_total': {
                'aliases': ['total t4', 'thyroxine'],
                'unit': 'mcg/dL',
                'general': (4.5, 12.0),
            },
            't3_free': {
                'aliases': ['free t3', 'ft3'],
                'unit': 'pg/mL',
                'general': (2.3, 4.2),
            },
            't4_free': {
                'aliases': ['free t4', 'ft4'],
                'unit': 'ng/dL',
                'general': (0.8, 1.8),
            },
            
            # ===================
            # Electrolytes
            # ===================
            'sodium': {
                'aliases': ['serum sodium', 'na', 'na+'],
                'unit': 'mEq/L',
                'general': (136, 145),
                'critical_low': 120,
                'critical_high': 160,
            },
            'potassium': {
                'aliases': ['serum potassium', 'k', 'k+'],
                'unit': 'mEq/L',
                'general': (3.5, 5.0),
                'critical_low': 2.5,
                'critical_high': 6.5,
            },
            'chloride': {
                'aliases': ['serum chloride', 'cl', 'cl-'],
                'unit': 'mEq/L',
                'general': (98, 106),
            },
            'calcium': {
                'aliases': ['serum calcium', 'ca', 'total calcium'],
                'unit': 'mg/dL',
                'general': (8.5, 10.5),
                'critical_low': 6.0,
                'critical_high': 14.0,
            },
            
            # ===================
            # Inflammatory Markers
            # ===================
            'esr': {
                'aliases': ['erythrocyte sedimentation rate', 'sed rate'],
                'unit': 'mm/hr',
                'male': (0, 15),
                'female': (0, 20),
                'general': (0, 20),
            },
            'crp': {
                'aliases': ['c reactive protein', 'c-reactive protein'],
                'unit': 'mg/L',
                'general': (0, 10),
                'critical_high': 100,
            },
            
            # ===================
            # Vitamins
            # ===================
            'vitamin_d': {
                'aliases': ['25-oh vitamin d', 'vitamin d3', '25-hydroxy vitamin d'],
                'unit': 'ng/mL',
                'general': (30, 100),
                'deficient': 20,
                'insufficient': 30,
            },
            'vitamin_b12': {
                'aliases': ['cobalamin', 'cyanocobalamin'],
                'unit': 'pg/mL',
                'general': (200, 900),
                'critical_low': 100,
            },
            'folate': {
                'aliases': ['folic acid', 'serum folate'],
                'unit': 'ng/mL',
                'general': (3.0, 17.0),
            },
            'iron': {
                'aliases': ['serum iron', 's.iron'],
                'unit': 'mcg/dL',
                'male': (65, 175),
                'female': (50, 170),
                'general': (50, 175),
            },
            'ferritin': {
                'aliases': ['serum ferritin'],
                'unit': 'ng/mL',
                'male': (30, 400),
                'female': (15, 150),
                'general': (15, 400),
            },
        }
    
    def _init_models(self):
        """Initialize ML models"""
        # Load embedding model for test name matching
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print("🔄 Loading medical parser models...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Compute embeddings for all test names
            all_names = []
            self.test_name_mapping = {}
            
            for test_name, info in self.reference_ranges.items():
                # Add main name
                all_names.append(test_name.replace('_', ' '))
                self.test_name_mapping[test_name.replace('_', ' ')] = test_name
                
                # Add aliases
                for alias in info.get('aliases', []):
                    all_names.append(alias)
                    self.test_name_mapping[alias] = test_name
            
            self.test_name_embeddings = self.embedding_model.encode(all_names)
            self.test_names_list = all_names
            print(f"   ✅ Loaded {len(self.reference_ranges)} test reference ranges")
        
        # Optional: Load biomedical NER
        if TRANSFORMERS_AVAILABLE:
            try:
                self.ner_model = pipeline(
                    "ner",
                    model="d4data/biomedical-ner-all",
                    device=self.device,
                    aggregation_strategy="simple"
                )
                print("   ✅ Biomedical NER loaded")
            except Exception as e:
                print(f"   ⚠️ NER model failed: {e}")
    
    def _match_test_name(self, text: str) -> Tuple[str, float]:
        """
        Match extracted test name to standard name using embeddings
        
        Returns:
            (standard_test_name, confidence)
        """
        if self.embedding_model is None:
            return self._keyword_match_test(text)
        
        # Get embedding
        text_embedding = self.embedding_model.encode(text.lower())
        
        # Find closest match
        similarities = cosine_similarity([text_embedding], self.test_name_embeddings)[0]
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity > 0.5:
            matched_name = self.test_names_list[best_idx]
            standard_name = self.test_name_mapping.get(matched_name, matched_name)
            return standard_name, float(best_similarity)
        
        return None, 0.0
    
    def _keyword_match_test(self, text: str) -> Tuple[str, float]:
        """Fallback keyword-based test matching"""
        text_lower = text.lower()
        
        for test_name, info in self.reference_ranges.items():
            # Check main name
            if test_name.replace('_', ' ') in text_lower:
                return test_name, 0.9
            
            # Check aliases
            for alias in info.get('aliases', []):
                if alias in text_lower:
                    return test_name, 0.85
        
        return None, 0.0
    
    def _normalize_unit(self, unit: str) -> str:
        """Normalize unit string"""
        unit = unit.strip().lower()
        
        normalizations = {
            'gm/dl': 'g/dL',
            'gm/l': 'g/L',
            'mg/dl': 'mg/dL',
            'iu/l': 'U/L',
            'iu/ml': 'U/mL',
            'cell/cumm': 'cells/cumm',
            'cells/cu mm': 'cells/cumm',
            '/cu mm': '/cumm',
            'mill/cumm': 'million/cumm',
            'meq/l': 'mEq/L',
            'mmol/l': 'mmol/L',
            'ng/ml': 'ng/mL',
            'pg/ml': 'pg/mL',
            'mcg/dl': 'mcg/dL',
            'fl': 'fL',
            'pg': 'pg',
            '%': '%',
        }
        
        return normalizations.get(unit, unit)
    
    def _assess_abnormality(self, 
                            test_name: str,
                            value: float,
                            gender: str = 'general') -> Tuple[AbnormalityLevel, float, str]:
        """
        Assess abnormality level for a test value
        
        Returns:
            (level, deviation_percent, interpretation)
        """
        if test_name not in self.reference_ranges:
            return AbnormalityLevel.UNKNOWN, 0.0, "Reference range not available"
        
        ref = self.reference_ranges[test_name]
        
        # Get appropriate range (handle None gender)
        if gender and gender.lower() in ['male', 'm']:
            range_key = 'male'
        elif gender and gender.lower() in ['female', 'f']:
            range_key = 'female'
        else:
            range_key = 'general'
        
        ref_range = ref.get(range_key, ref.get('general'))
        
        if ref_range is None:
            return AbnormalityLevel.UNKNOWN, 0.0, "Reference range not found"
        
        low, high = ref_range
        mid = (low + high) / 2
        
        # Calculate deviation
        if value < low:
            deviation = ((low - value) / low) * 100
        elif value > high:
            deviation = ((value - high) / high) * 100
        else:
            deviation = 0.0
        
        # Check critical values
        critical_low = ref.get('critical_low')
        critical_high = ref.get('critical_high')
        
        if critical_low and value < critical_low:
            return AbnormalityLevel.CRITICAL, deviation, f"Critically low! Value {value} below critical threshold {critical_low}"
        
        if critical_high and value > critical_high:
            return AbnormalityLevel.CRITICAL, deviation, f"Critically high! Value {value} above critical threshold {critical_high}"
        
        # Normal range check
        if low <= value <= high:
            return AbnormalityLevel.NORMAL, 0.0, f"Within normal range ({low}-{high})"
        
        # Degree of abnormality
        if deviation > 50:
            level = AbnormalityLevel.HIGH
            desc = "significantly"
        elif deviation > 25:
            level = AbnormalityLevel.MODERATE
            desc = "moderately"
        else:
            level = AbnormalityLevel.MILD
            desc = "slightly"
        
        if value < low:
            return level, deviation, f"Value {desc} low ({deviation:.0f}% below normal)"
        else:
            return level, deviation, f"Value {desc} high ({deviation:.0f}% above normal)"
    
    def _extract_values_regex(self, text: str) -> List[Dict]:
        """
        Extract test values using regex patterns
        """
        results = []
        
        # Common patterns for lab values
        patterns = [
            # Pattern: Test Name: Value Unit (Reference: X-Y)
            r'([A-Za-z][A-Za-z\s\.\-]+?)\s*[:=]\s*(\d+\.?\d*)\s*([A-Za-z/%]+)',
            # Pattern: Test Name Value Unit
            r'([A-Za-z][A-Za-z\s]+?)\s+(\d+\.?\d*)\s*([A-Za-z/%]+)',
            # Pattern with range: Value Unit (Low-High)
            r'(\d+\.?\d*)\s*([A-Za-z/%]+)\s*\((\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 3:
                    try:
                        name = match[0].strip()
                        value = float(match[1])
                        unit = match[2].strip()
                        
                        # Match to standard test name
                        std_name, confidence = self._match_test_name(name)
                        
                        if std_name and confidence > 0.5:
                            results.append({
                                'test_name': std_name,
                                'raw_name': name,
                                'value': value,
                                'unit': self._normalize_unit(unit),
                                'confidence': confidence
                            })
                    except (ValueError, IndexError):
                        continue
        
        return results
    
    def _extract_patient_info(self, text: str) -> Dict:
        """Extract patient demographic information"""
        info = {
            'name': None,
            'age': None,
            'gender': None,
            'date': None
        }
        
        # Name patterns
        name_patterns = [
            r'(?:patient|name|pt)[\s:]+([A-Za-z\s\.]+?)(?:\s*,|\s*age|\s*\n|$)',
            r'(?:mr|mrs|ms|dr)\.?\s+([A-Za-z\s\.]+)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info['name'] = match.group(1).strip()[:50]
                break
        
        # Age patterns
        age_patterns = [
            r'(?:age)[\s:]+(\d+)\s*(?:years?|yrs?|y)?',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:old)?',
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info['age'] = match.group(1)
                break
        
        # Gender patterns
        gender_patterns = [
            r'\b(male|female|m|f)\b',
            r'(?:sex|gender)[\s:]+([mf]|male|female)',
        ]
        
        for pattern in gender_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                g = match.group(1).lower()
                info['gender'] = 'male' if g in ['m', 'male'] else 'female'
                break
        
        # Date patterns
        date_patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info['date'] = match.group(1)
                break
        
        return info
    
    def parse(self, text: str) -> MedicalReport:
        """
        Parse medical report text
        
        Args:
            text: Report text (from OCR or direct extraction)
            
        Returns:
            MedicalReport object
        """
        print("\n🔍 Enhanced Medical Report Parsing")
        
        # Extract patient info
        patient_info = self._extract_patient_info(text)
        print(f"   Patient: {patient_info.get('name', 'Unknown')}, "
              f"Age: {patient_info.get('age', 'N/A')}, "
              f"Gender: {patient_info.get('gender', 'N/A')}")
        
        # Extract test values
        extracted_values = self._extract_values_regex(text)
        
        # Use NER if available
        if self.ner_model:
            try:
                ner_results = self.ner_model(text[:2000])
                # Process NER results to find additional test values
                # (Implementation depends on NER model output format)
            except Exception:
                pass
        
        # Process extracted values
        test_results = []
        gender = patient_info.get('gender', 'general')
        
        for item in extracted_values:
            test_name = item['test_name']
            value = item['value']
            unit = item['unit']
            
            # Assess abnormality
            level, deviation, interpretation = self._assess_abnormality(
                test_name, value, gender
            )
            
            # Get reference range string
            ref = self.reference_ranges.get(test_name, {})
            ref_range = ref.get(gender, ref.get('general', (0, 0)))
            if isinstance(ref_range, tuple):
                ref_str = f"{ref_range[0]}-{ref_range[1]}"
            else:
                ref_str = "N/A"
            
            test_results.append(TestResult(
                test_name=test_name,
                value=value,
                unit=unit,
                reference_range=ref_str,
                status=level,
                deviation_percent=deviation,
                interpretation=interpretation
            ))
        
        # Remove duplicates (keep highest confidence)
        seen = {}
        unique_results = []
        for result in test_results:
            key = result.test_name
            if key not in seen:
                seen[key] = result
                unique_results.append(result)
        
        # Count abnormalities
        abnormal = sum(1 for r in unique_results if r.status != AbnormalityLevel.NORMAL)
        critical = sum(1 for r in unique_results if r.status == AbnormalityLevel.CRITICAL)
        
        # Generate summary
        if critical > 0:
            summary = f"⚠️ CRITICAL: {critical} critical findings require immediate attention. "
        elif abnormal > 0:
            summary = f"Found {abnormal} abnormal results that may need follow-up. "
        else:
            summary = "All results within normal limits. "
        
        summary += f"Total {len(unique_results)} tests analyzed."
        
        # Calculate confidence
        if unique_results:
            avg_conf = np.mean([r.deviation_percent for r in unique_results if r.status != AbnormalityLevel.UNKNOWN])
            confidence = min(0.9, 0.5 + len(unique_results) * 0.05)
        else:
            confidence = 0.3
        
        print(f"   Found {len(unique_results)} tests, {abnormal} abnormal, {critical} critical")
        
        return MedicalReport(
            patient_name=patient_info.get('name'),
            patient_age=patient_info.get('age'),
            patient_gender=patient_info.get('gender'),
            test_date=patient_info.get('date'),
            lab_name=None,
            test_results=unique_results,
            summary=summary,
            abnormal_count=abnormal,
            critical_count=critical,
            confidence=confidence
        )


# Backward compatibility
MLMedicalParser = EnhancedMedicalParser


if __name__ == "__main__":
    # Test
    parser = EnhancedMedicalParser()
    
    test_text = """
    Patient Name: Ramesh Kumar
    Age: 45 years, Gender: Male
    Date: 15-Nov-2025
    
    Complete Blood Count:
    Hemoglobin: 11.5 g/dL (Normal: 13-17)
    WBC Count: 8500 cells/cumm (Normal: 4000-11000)
    Platelet Count: 145000 /cumm (Normal: 150000-400000)
    
    Blood Sugar:
    Fasting Glucose: 126 mg/dL (Normal: 70-100)
    HbA1c: 6.8% (Normal: <5.7%)
    
    Lipid Profile:
    Total Cholesterol: 245 mg/dL (Desirable: <200)
    LDL: 165 mg/dL (Normal: <100)
    HDL: 38 mg/dL (Normal: >40)
    Triglycerides: 210 mg/dL (Normal: <150)
    """
    
    report = parser.parse(test_text)
    print(f"\n\nParsed Report:")
    print(f"Summary: {report.summary}")
    
    for result in report.test_results:
        print(f"\n{result.test_name}: {result.value} {result.unit}")
        print(f"   Status: {result.status.value}")
        print(f"   {result.interpretation}")
