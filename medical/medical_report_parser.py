"""
Medical Report Parser Module
============================

Extracts structured data from medical lab report OCR text:
- Test names (with standardization)
- Values (numeric)
- Units (g/dL, mg/dL, cells/cumm, etc.)
- Reference ranges
- Flags (Normal, High, Low, Critical)

Handles common Indian lab report formats from:
- Apollo, Thyrocare, SRL, Dr. Lal PathLabs, etc.

Author: PlainSense Team
Date: November 2025
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class ResultFlag(Enum):
    """Test result status flags"""
    NORMAL = "normal"
    LOW = "low"
    HIGH = "high"
    CRITICAL_LOW = "critical_low"
    CRITICAL_HIGH = "critical_high"
    UNKNOWN = "unknown"


@dataclass
class TestResult:
    """Structured test result"""
    test_name: str
    test_name_standardized: str
    value: Optional[float]
    value_raw: str
    unit: str
    reference_range: str
    ref_low: Optional[float]
    ref_high: Optional[float]
    flag: ResultFlag
    flag_source: str  # 'ocr' (from report), 'calculated', 'unknown'
    line_number: int
    raw_text: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['flag'] = self.flag.value
        return result
    
    def is_abnormal(self) -> bool:
        """Check if result is outside normal range"""
        return self.flag in [ResultFlag.LOW, ResultFlag.HIGH, 
                            ResultFlag.CRITICAL_LOW, ResultFlag.CRITICAL_HIGH]
    
    def is_critical(self) -> bool:
        """Check if result is critically abnormal"""
        return self.flag in [ResultFlag.CRITICAL_LOW, ResultFlag.CRITICAL_HIGH]


class MedicalReportParser:
    """
    Parser for extracting structured data from medical lab reports
    """
    
    def __init__(self):
        """Initialize parser with test name mappings and patterns"""
        
        # Test name standardization (aliases -> standard name)
        self.test_name_map = {
            # Hemoglobin variants
            'hb': 'Hemoglobin',
            'hgb': 'Hemoglobin',
            'haemoglobin': 'Hemoglobin',
            'hemoglobin': 'Hemoglobin',
            
            # Blood Sugar variants
            'fbs': 'Fasting Blood Sugar',
            'fasting blood sugar': 'Fasting Blood Sugar',
            'fasting glucose': 'Fasting Blood Sugar',
            'fasting blood glucose': 'Fasting Blood Sugar',
            'glucose fasting': 'Fasting Blood Sugar',
            'ppbs': 'Post Prandial Blood Sugar',
            'post prandial blood sugar': 'Post Prandial Blood Sugar',
            'pp blood sugar': 'Post Prandial Blood Sugar',
            'rbs': 'Random Blood Sugar',
            'random blood sugar': 'Random Blood Sugar',
            'blood sugar random': 'Random Blood Sugar',
            'hba1c': 'HbA1c',
            'glycated hemoglobin': 'HbA1c',
            'glycosylated hemoglobin': 'HbA1c',
            
            # Complete Blood Count
            'rbc': 'Red Blood Cell Count',
            'rbc count': 'Red Blood Cell Count',
            'red blood cell count': 'Red Blood Cell Count',
            'red blood cells': 'Red Blood Cell Count',
            'wbc': 'White Blood Cell Count',
            'wbc count': 'White Blood Cell Count',
            'white blood cell count': 'White Blood Cell Count',
            'white blood cells': 'White Blood Cell Count',
            'tlc': 'Total Leucocyte Count',
            'total leucocyte count': 'Total Leucocyte Count',
            'total leukocyte count': 'Total Leucocyte Count',
            'platelet': 'Platelet Count',
            'platelet count': 'Platelet Count',
            'platelets': 'Platelet Count',
            'plt': 'Platelet Count',
            
            # RBC Indices
            'mcv': 'Mean Corpuscular Volume',
            'mean corpuscular volume': 'Mean Corpuscular Volume',
            'mch': 'Mean Corpuscular Hemoglobin',
            'mean corpuscular hemoglobin': 'Mean Corpuscular Hemoglobin',
            'mchc': 'Mean Corpuscular Hemoglobin Concentration',
            'mean corpuscular hemoglobin concentration': 'Mean Corpuscular Hemoglobin Concentration',
            'rdw': 'Red Cell Distribution Width',
            'rdw-cv': 'Red Cell Distribution Width',
            'red cell distribution width': 'Red Cell Distribution Width',
            'pcv': 'Packed Cell Volume',
            'packed cell volume': 'Packed Cell Volume',
            'hematocrit': 'Packed Cell Volume',
            'haematocrit': 'Packed Cell Volume',
            'hct': 'Packed Cell Volume',
            
            # Differential Count
            'neutrophils': 'Neutrophils',
            'neutrophil': 'Neutrophils',
            'lymphocytes': 'Lymphocytes',
            'lymphocyte': 'Lymphocytes',
            'monocytes': 'Monocytes',
            'monocyte': 'Monocytes',
            'eosinophils': 'Eosinophils',
            'eosinophil': 'Eosinophils',
            'basophils': 'Basophils',
            'basophil': 'Basophils',
            
            # Kidney Function
            'creatinine': 'Serum Creatinine',
            'serum creatinine': 'Serum Creatinine',
            's. creatinine': 'Serum Creatinine',
            'urea': 'Blood Urea',
            'blood urea': 'Blood Urea',
            'bun': 'Blood Urea Nitrogen',
            'blood urea nitrogen': 'Blood Urea Nitrogen',
            'uric acid': 'Uric Acid',
            'serum uric acid': 'Uric Acid',
            'egfr': 'eGFR',
            'gfr': 'eGFR',
            'estimated gfr': 'eGFR',
            
            # Liver Function
            'sgot': 'SGOT (AST)',
            'ast': 'SGOT (AST)',
            'aspartate aminotransferase': 'SGOT (AST)',
            'sgpt': 'SGPT (ALT)',
            'alt': 'SGPT (ALT)',
            'alanine aminotransferase': 'SGPT (ALT)',
            'bilirubin': 'Total Bilirubin',
            'total bilirubin': 'Total Bilirubin',
            'direct bilirubin': 'Direct Bilirubin',
            'indirect bilirubin': 'Indirect Bilirubin',
            'alkaline phosphatase': 'Alkaline Phosphatase',
            'alp': 'Alkaline Phosphatase',
            'albumin': 'Serum Albumin',
            'serum albumin': 'Serum Albumin',
            'globulin': 'Serum Globulin',
            'total protein': 'Total Protein',
            'serum protein': 'Total Protein',
            'ggtp': 'GGTP',
            'ggt': 'GGTP',
            'gamma gt': 'GGTP',
            
            # Lipid Profile
            'cholesterol': 'Total Cholesterol',
            'total cholesterol': 'Total Cholesterol',
            'triglycerides': 'Triglycerides',
            'tg': 'Triglycerides',
            'hdl': 'HDL Cholesterol',
            'hdl cholesterol': 'HDL Cholesterol',
            'hdl-c': 'HDL Cholesterol',
            'ldl': 'LDL Cholesterol',
            'ldl cholesterol': 'LDL Cholesterol',
            'ldl-c': 'LDL Cholesterol',
            'vldl': 'VLDL Cholesterol',
            'vldl cholesterol': 'VLDL Cholesterol',
            
            # Thyroid
            'tsh': 'TSH',
            'thyroid stimulating hormone': 'TSH',
            't3': 'T3',
            'triiodothyronine': 'T3',
            't4': 'T4',
            'thyroxine': 'T4',
            'free t3': 'Free T3',
            'ft3': 'Free T3',
            'free t4': 'Free T4',
            'ft4': 'Free T4',
            
            # Vitamins & Minerals
            'vitamin d': 'Vitamin D',
            'vit d': 'Vitamin D',
            '25-oh vitamin d': 'Vitamin D',
            'vitamin b12': 'Vitamin B12',
            'vit b12': 'Vitamin B12',
            'b12': 'Vitamin B12',
            'iron': 'Serum Iron',
            'serum iron': 'Serum Iron',
            'ferritin': 'Serum Ferritin',
            'serum ferritin': 'Serum Ferritin',
            'calcium': 'Serum Calcium',
            'serum calcium': 'Serum Calcium',
            
            # Electrolytes
            'sodium': 'Serum Sodium',
            'serum sodium': 'Serum Sodium',
            'na': 'Serum Sodium',
            'potassium': 'Serum Potassium',
            'serum potassium': 'Serum Potassium',
            'k': 'Serum Potassium',
            'chloride': 'Serum Chloride',
            'serum chloride': 'Serum Chloride',
            'cl': 'Serum Chloride',
            
            # Urine
            'urine sugar': 'Urine Sugar',
            'urine glucose': 'Urine Sugar',
            'urine protein': 'Urine Protein',
            'urine albumin': 'Urine Albumin',
            'specific gravity': 'Urine Specific Gravity',
            'urine ph': 'Urine pH',
            
            # Cardiac
            'troponin': 'Troponin',
            'troponin i': 'Troponin I',
            'troponin t': 'Troponin T',
            'cpk': 'CPK',
            'creatine kinase': 'CPK',
            'ck-mb': 'CK-MB',
            'bnp': 'BNP',
            'nt-probnp': 'NT-proBNP',
            
            # Coagulation
            'pt': 'Prothrombin Time',
            'prothrombin time': 'Prothrombin Time',
            'inr': 'INR',
            'aptt': 'aPTT',
            'activated partial thromboplastin time': 'aPTT',
            'd-dimer': 'D-Dimer',
            'd dimer': 'D-Dimer',
            
            # Inflammation
            'esr': 'ESR',
            'erythrocyte sedimentation rate': 'ESR',
            'crp': 'C-Reactive Protein',
            'c-reactive protein': 'C-Reactive Protein',
            'c reactive protein': 'C-Reactive Protein',
        }
        
        # Unit standardization
        self.unit_map = {
            'g/dl': 'g/dL',
            'gm/dl': 'g/dL',
            'gm%': 'g/dL',
            'g%': 'g/dL',
            'mg/dl': 'mg/dL',
            'mg%': 'mg/dL',
            'cells/cumm': 'cells/cumm',
            'cells/cmm': 'cells/cumm',
            '/cumm': 'cells/cumm',
            '/cmm': 'cells/cumm',
            'mill/cumm': 'million/cumm',
            'million/cumm': 'million/cumm',
            'mill/cmm': 'million/cumm',
            'lakhs/cumm': 'lakhs/cumm',
            'lakh/cumm': 'lakhs/cumm',
            'thou/cumm': 'thousand/cumm',
            'thousand/cumm': 'thousand/cumm',
            '/ul': '/µL',
            '/μl': '/µL',
            'k/ul': 'K/µL',
            'k/μl': 'K/µL',
            'm/ul': 'M/µL',
            'm/μl': 'M/µL',
            'fl': 'fL',
            'pg': 'pg',
            'g/dl': 'g/dL',
            '%': '%',
            'iu/l': 'IU/L',
            'u/l': 'U/L',
            'miu/ml': 'mIU/mL',
            'miu/l': 'mIU/L',
            'uiu/ml': 'µIU/mL',
            'ng/ml': 'ng/mL',
            'ng/dl': 'ng/dL',
            'pg/ml': 'pg/mL',
            'meq/l': 'mEq/L',
            'mmol/l': 'mmol/L',
            'mm/hr': 'mm/hr',
            'mm/1st hr': 'mm/hr',
            'seconds': 'seconds',
            'sec': 'seconds',
            'ratio': 'ratio',
        }
        
        # Patterns for extracting test results
        self._compile_patterns()
        
        # Critical value thresholds (test_name -> (critical_low, critical_high))
        self.critical_thresholds = {
            'Hemoglobin': (7.0, 20.0),
            'Fasting Blood Sugar': (50, 400),
            'Serum Creatinine': (None, 10.0),
            'Serum Potassium': (2.5, 6.5),
            'Serum Sodium': (120, 160),
            'Platelet Count': (50000, 1000000),
            'White Blood Cell Count': (2000, 30000),
            'INR': (None, 5.0),
            'Troponin': (None, 0.4),
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for extraction"""
        
        # Pattern for numeric values (handles decimals, negative, spaces)
        self.value_pattern = r'[-+]?\d+\.?\d*'
        
        # Pattern for units (comprehensive)
        self.unit_pattern = r'''
            (?:
                g/d[lL]|gm/d[lL]|gm%|g%|                    # Hemoglobin units
                mg/d[lL]|mg%|                               # Common units
                cells?/c?u?mm|/c?u?mm|                      # Cell counts
                mill(?:ion)?/c?u?mm|lakhs?/c?u?mm|          # Million/Lakh counts
                thou(?:sand)?/c?u?mm|                       # Thousand counts
                [KkMm]?/[uμµ][Ll]|                          # Per microliter
                f[Ll]|p[gG]|                                # fL, pg
                %|                                          # Percentage
                [IU]U/[LlmM][Ll]?|U/[Ll]|                   # Enzyme units
                m?[IU]U/m[Ll]|[uμµ]IU/m[Ll]|                # Hormone units
                n[gG]/[md][Ll]|p[gG]/m[Ll]|                 # ng/mL, pg/mL
                m?[Ee]q/[Ll]|mmol/[Ll]|                     # Electrolytes
                mm/(?:1st\s*)?h(?:ou)?r|                    # ESR
                sec(?:onds?)?|                              # Time
                ratio                                       # Ratio
            )
        '''
        
        # Pattern for reference ranges
        self.ref_range_patterns = [
            # 12.0 - 16.0 or 12.0-16.0
            rf'({self.value_pattern})\s*[-–—to]+\s*({self.value_pattern})',
            # <100 or >50
            rf'([<>≤≥])\s*({self.value_pattern})',
            # (12.0-16.0) with parentheses
            rf'\(\s*({self.value_pattern})\s*[-–—to]+\s*({self.value_pattern})\s*\)',
            # Ref: 12.0-16.0
            rf'[Rr]ef(?:erence)?[:\s]*({self.value_pattern})\s*[-–—to]+\s*({self.value_pattern})',
        ]
        
        # Pattern for flags in OCR text
        self.flag_patterns = [
            (r'\b(LOW|L)\b', ResultFlag.LOW),
            (r'\b(HIGH|H)\b', ResultFlag.HIGH),
            (r'\b(NORMAL|N)\b', ResultFlag.NORMAL),
            (r'\b(CRITICAL|CRIT|PANIC)\b', ResultFlag.CRITICAL_HIGH),  # Will determine high/low from value
            (r'\*+', ResultFlag.HIGH),  # Asterisks often indicate abnormal
            (r'↓|⬇|▼', ResultFlag.LOW),
            (r'↑|⬆|▲', ResultFlag.HIGH),
        ]
    
    def standardize_test_name(self, name: str) -> str:
        """Standardize test name to canonical form"""
        # Clean and lowercase
        cleaned = name.strip().lower()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'[:\-_]+$', '', cleaned)
        
        # Look up in mapping
        if cleaned in self.test_name_map:
            return self.test_name_map[cleaned]
        
        # Try partial matches
        for key, standard in self.test_name_map.items():
            if key in cleaned or cleaned in key:
                return standard
        
        # Return original with title case if not found
        return name.strip().title()
    
    def standardize_unit(self, unit: str) -> str:
        """Standardize unit to canonical form"""
        cleaned = unit.strip().lower()
        return self.unit_map.get(cleaned, unit.strip())
    
    def parse_reference_range(self, text: str) -> Tuple[Optional[float], Optional[float], str]:
        """
        Parse reference range from text
        
        Returns:
            Tuple of (low, high, raw_range_string)
        """
        for pattern in self.ref_range_patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                
                if len(groups) == 2:
                    if groups[0] in '<>≤≥':
                        # Single bound (<100 or >50)
                        value = float(groups[1])
                        if groups[0] in '<≤':
                            return (None, value, match.group(0))
                        else:
                            return (value, None, match.group(0))
                    else:
                        # Range (12.0-16.0)
                        try:
                            low = float(groups[0])
                            high = float(groups[1])
                            return (low, high, match.group(0))
                        except ValueError:
                            pass
        
        return (None, None, "")
    
    def extract_flag_from_text(self, text: str) -> Tuple[ResultFlag, str]:
        """Extract result flag from OCR text"""
        text_upper = text.upper()
        
        for pattern, flag in self.flag_patterns:
            if re.search(pattern, text_upper):
                return (flag, 'ocr')
        
        return (ResultFlag.UNKNOWN, 'unknown')
    
    def calculate_flag(self, value: Optional[float], ref_low: Optional[float], 
                       ref_high: Optional[float], test_name: str) -> ResultFlag:
        """Calculate flag based on value and reference range"""
        if value is None:
            return ResultFlag.UNKNOWN
        
        # Check critical thresholds first
        std_name = self.standardize_test_name(test_name)
        if std_name in self.critical_thresholds:
            crit_low, crit_high = self.critical_thresholds[std_name]
            if crit_low is not None and value < crit_low:
                return ResultFlag.CRITICAL_LOW
            if crit_high is not None and value > crit_high:
                return ResultFlag.CRITICAL_HIGH
        
        # Check against reference range
        if ref_low is not None and value < ref_low:
            return ResultFlag.LOW
        if ref_high is not None and value > ref_high:
            return ResultFlag.HIGH
        if ref_low is not None or ref_high is not None:
            return ResultFlag.NORMAL
        
        return ResultFlag.UNKNOWN
    
    def parse_line(self, line: str, line_number: int) -> Optional[TestResult]:
        """
        Parse a single line to extract test result
        
        Args:
            line: OCR text line
            line_number: Line number in document
            
        Returns:
            TestResult if successful, None otherwise
        """
        # Skip empty lines or headers
        if not line.strip() or len(line.strip()) < 5:
            return None
        
        # Skip common header/footer patterns
        skip_patterns = [
            r'^page\s+\d+',
            r'^date[:\s]',
            r'^name[:\s]',
            r'^age[:\s]',
            r'^sex[:\s]',
            r'^sample',
            r'^collected',
            r'^reported',
            r'^doctor',
            r'^ref\.\s*(?:by|doctor)',
            r'^\*+$',
            r'^[-=_]+$',
        ]
        
        line_lower = line.lower().strip()
        for pattern in skip_patterns:
            if re.match(pattern, line_lower):
                return None
        
        # Try to extract components
        # Pattern: TestName ... Value Unit (RefRange) Flag
        
        # Extract value (look for numbers)
        value_matches = list(re.finditer(rf'({self.value_pattern})', line))
        if not value_matches:
            return None
        
        # Extract unit
        unit_match = re.search(self.unit_pattern, line, re.VERBOSE | re.IGNORECASE)
        unit = unit_match.group(0) if unit_match else ""
        
        # Extract reference range
        ref_low, ref_high, ref_range_str = self.parse_reference_range(line)
        
        # Extract flag from text
        ocr_flag, flag_source = self.extract_flag_from_text(line)
        
        # Determine test name (usually at the start, before numbers)
        first_value_pos = value_matches[0].start()
        test_name_raw = line[:first_value_pos].strip()
        
        # Clean test name
        test_name_raw = re.sub(r'[:\-_]+$', '', test_name_raw).strip()
        test_name_raw = re.sub(r'\s+', ' ', test_name_raw)
        
        if len(test_name_raw) < 2:
            return None
        
        # Get the primary value (usually the first or second number)
        # Skip if first number looks like it's part of the test name (e.g., "Vitamin B12")
        value_raw = ""
        value = None
        
        for match in value_matches:
            # Check if this number is part of test name
            if match.start() < len(test_name_raw) + 5:
                # Might be part of test name, check context
                before = line[max(0, match.start()-3):match.start()]
                if re.search(r'[a-zA-Z]$', before):
                    continue  # Skip, likely part of name like "B12"
            
            try:
                value_raw = match.group(1)
                value = float(value_raw)
                break
            except ValueError:
                continue
        
        if value is None:
            return None
        
        # Calculate flag if not detected from OCR
        if ocr_flag == ResultFlag.UNKNOWN:
            calculated_flag = self.calculate_flag(value, ref_low, ref_high, test_name_raw)
            flag = calculated_flag
            flag_source = 'calculated' if calculated_flag != ResultFlag.UNKNOWN else 'unknown'
        else:
            flag = ocr_flag
        
        return TestResult(
            test_name=test_name_raw,
            test_name_standardized=self.standardize_test_name(test_name_raw),
            value=value,
            value_raw=value_raw,
            unit=self.standardize_unit(unit),
            reference_range=ref_range_str,
            ref_low=ref_low,
            ref_high=ref_high,
            flag=flag,
            flag_source=flag_source,
            line_number=line_number,
            raw_text=line.strip()
        )
    
    def parse_report(self, text: str) -> Dict:
        """
        Parse complete medical report text
        
        Args:
            text: Full OCR text from medical report
            
        Returns:
            Dictionary with parsed results and metadata
        """
        lines = text.split('\n')
        results = []
        
        for i, line in enumerate(lines, 1):
            result = self.parse_line(line, i)
            if result:
                results.append(result)
        
        # Categorize results
        abnormal_results = [r for r in results if r.is_abnormal()]
        critical_results = [r for r in results if r.is_critical()]
        
        return {
            'total_tests': len(results),
            'abnormal_count': len(abnormal_results),
            'critical_count': len(critical_results),
            'results': [r.to_dict() for r in results],
            'abnormal_results': [r.to_dict() for r in abnormal_results],
            'critical_results': [r.to_dict() for r in critical_results],
            'summary': self._generate_summary(results, abnormal_results, critical_results)
        }
    
    def _generate_summary(self, results: List[TestResult], 
                          abnormal: List[TestResult],
                          critical: List[TestResult]) -> str:
        """Generate human-readable summary"""
        lines = []
        
        lines.append(f"📊 Total tests extracted: {len(results)}")
        
        if critical:
            lines.append(f"\n🚨 CRITICAL VALUES ({len(critical)}):")
            for r in critical:
                lines.append(f"   • {r.test_name_standardized}: {r.value} {r.unit} ({r.flag.value.upper()})")
        
        if abnormal:
            non_critical_abnormal = [r for r in abnormal if not r.is_critical()]
            if non_critical_abnormal:
                lines.append(f"\n⚠️  ABNORMAL VALUES ({len(non_critical_abnormal)}):")
                for r in non_critical_abnormal:
                    flag_text = "↓ LOW" if r.flag == ResultFlag.LOW else "↑ HIGH"
                    lines.append(f"   • {r.test_name_standardized}: {r.value} {r.unit} {flag_text}")
        
        if not abnormal:
            lines.append("\n✅ All values within normal range")
        
        return '\n'.join(lines)
    
    def parse_and_simplify(self, text: str) -> Dict:
        """
        Parse report and generate simplified explanations
        
        Args:
            text: OCR text from medical report
            
        Returns:
            Dictionary with parsed data and simplified explanations
        """
        parsed = self.parse_report(text)
        
        # Add simplified explanations for each result
        for result in parsed['results']:
            result['explanation'] = self._explain_result(result)
        
        return parsed
    
    def _explain_result(self, result: Dict) -> str:
        """Generate simple explanation for a test result"""
        name = result['test_name_standardized']
        value = result['value']
        unit = result['unit']
        flag = result['flag']
        ref = result['reference_range']
        
        # Base explanation
        explanation = f"{name} is {value} {unit}."
        
        if ref:
            explanation += f" Normal range is {ref}."
        
        # Add flag-specific explanation
        if flag == 'low':
            explanation += f" Your result is LOWER than normal."
        elif flag == 'high':
            explanation += f" Your result is HIGHER than normal."
        elif flag == 'critical_low':
            explanation += f" ⚠️ Your result is CRITICALLY LOW. Please consult a doctor immediately."
        elif flag == 'critical_high':
            explanation += f" ⚠️ Your result is CRITICALLY HIGH. Please consult a doctor immediately."
        elif flag == 'normal':
            explanation += f" This is within normal range."
        
        return explanation


# =============================================================================
# Convenience Functions
# =============================================================================

def parse_medical_report(text: str) -> Dict:
    """
    Quick function to parse medical report text
    
    Args:
        text: OCR text from medical report
        
    Returns:
        Parsed results dictionary
    """
    parser = MedicalReportParser()
    return parser.parse_report(text)


def parse_and_explain(text: str) -> Dict:
    """
    Parse report and add simplified explanations
    
    Args:
        text: OCR text from medical report
        
    Returns:
        Results with explanations
    """
    parser = MedicalReportParser()
    return parser.parse_and_simplify(text)


# =============================================================================
# Main - Demo & Testing
# =============================================================================

if __name__ == '__main__':
    # Sample medical report text (simulating OCR output)
    sample_report = """
    COMPLETE BLOOD COUNT REPORT
    
    Patient Name: John Doe
    Age: 45 Years
    Sex: Male
    Date: 28-Nov-2025
    
    TEST                        RESULT      UNIT            REFERENCE RANGE     FLAG
    
    Hemoglobin                  10.2        g/dL            12.0 - 16.0         LOW
    Red Blood Cell Count        4.2         million/cumm    4.5 - 5.5           LOW
    White Blood Cell Count      12500       cells/cumm      4000 - 11000        HIGH
    Platelet Count              245000      cells/cumm      150000 - 400000     NORMAL
    
    Packed Cell Volume (PCV)    32          %               36 - 46             LOW
    MCV                         76          fL              80 - 100            LOW
    MCH                         24.3        pg              27 - 32             LOW
    MCHC                        31.9        g/dL            32 - 36             LOW
    
    DIFFERENTIAL COUNT:
    Neutrophils                 72          %               40 - 70             HIGH
    Lymphocytes                 20          %               20 - 40             NORMAL
    Monocytes                   5           %               2 - 8               NORMAL
    Eosinophils                 2           %               1 - 6               NORMAL
    Basophils                   1           %               0 - 1               NORMAL
    
    ESR                         45          mm/hr           0 - 15              HIGH
    
    BLOOD SUGAR:
    Fasting Blood Sugar         250         mg/dL           70 - 100            HIGH
    
    ---
    Dr. Smith
    Pathologist
    """
    
    print("="*70)
    print("MEDICAL REPORT PARSER - DEMO")
    print("="*70)
    
    parser = MedicalReportParser()
    result = parser.parse_and_simplify(sample_report)
    
    print(f"\n{result['summary']}")
    
    print("\n" + "="*70)
    print("DETAILED RESULTS:")
    print("="*70)
    
    for r in result['results'][:5]:  # Show first 5
        print(f"\n📋 {r['test_name_standardized']}")
        print(f"   Value: {r['value']} {r['unit']}")
        print(f"   Reference: {r['reference_range']}")
        print(f"   Status: {r['flag'].upper()}")
        print(f"   💡 {r['explanation']}")
    
    print("\n" + "="*70)
    print("JSON OUTPUT (sample):")
    print("="*70)
    import json
    print(json.dumps(result['results'][:2], indent=2))
