"""
Medical Dictionary / Assistant Module
=====================================

Provides explanations for:
- Medical test names and what they measure
- Conditions and diseases (symptoms, causes, treatments)
- Medical terminology
- Lab value interpretations

This assists users in understanding their medical reports.

Author: PlainSense Team
Date: November 2025
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

# Try to import LLM for enhanced explanations
try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class MedicalTerm:
    """Represents a medical term with its explanation"""
    term: str
    simple_name: str
    category: str  # 'test', 'condition', 'medication', 'procedure'
    description: str
    normal_range: Optional[str] = None
    unit: Optional[str] = None
    high_meaning: Optional[str] = None
    low_meaning: Optional[str] = None
    symptoms: Optional[List[str]] = None
    related_conditions: Optional[List[str]] = None


class MedicalDictionary:
    """
    Medical dictionary for explaining medical terms, tests, and conditions.
    Uses a combination of:
    1. Built-in medical knowledge base
    2. LLM for enhanced/custom explanations
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize the medical dictionary.
        
        Args:
            use_llm: Whether to use LLM for enhanced explanations
        """
        self.use_llm = use_llm and TRANSFORMERS_AVAILABLE
        self.llm_model = None
        self.llm_tokenizer = None
        
        # Initialize knowledge base
        self._init_knowledge_base()
        
        print("📚 Medical Dictionary initialized")
    
    def _init_knowledge_base(self):
        """Initialize the medical knowledge base"""
        
        # Lab Tests Dictionary
        self.lab_tests: Dict[str, MedicalTerm] = {
            # Blood Count
            'hemoglobin': MedicalTerm(
                term='Hemoglobin',
                simple_name='Oxygen carrier in blood',
                category='test',
                description='A protein in red blood cells that carries oxygen from your lungs to the rest of your body and returns carbon dioxide back to your lungs.',
                normal_range='12-17 g/dL (varies by age/gender)',
                unit='g/dL',
                high_meaning='May indicate dehydration, lung disease, or living at high altitude. Rarely, can indicate blood disorders.',
                low_meaning='Indicates anemia. You may feel tired, weak, or short of breath. Common causes: iron deficiency, blood loss, vitamin deficiency.',
                related_conditions=['Anemia', 'Polycythemia', 'Thalassemia']
            ),
            'rbc': MedicalTerm(
                term='Red Blood Cell Count',
                simple_name='Red blood cells',
                category='test',
                description='Measures the number of red blood cells in your blood. Red cells carry oxygen throughout your body.',
                normal_range='4.5-5.5 million/μL',
                unit='million cells/μL',
                high_meaning='May indicate dehydration, heart disease, or kidney problems.',
                low_meaning='May indicate anemia, bleeding, or bone marrow problems.',
                related_conditions=['Anemia', 'Polycythemia vera']
            ),
            'wbc': MedicalTerm(
                term='White Blood Cell Count',
                simple_name='Infection-fighting cells',
                category='test',
                description='Measures white blood cells that fight infections. High or low counts can indicate various conditions.',
                normal_range='4,000-11,000/μL',
                unit='cells/μL',
                high_meaning='Usually indicates infection, inflammation, or stress. Very high counts need immediate attention.',
                low_meaning='May indicate immune system problems, bone marrow issues, or effects of certain medications.',
                related_conditions=['Infections', 'Leukemia', 'Immune disorders']
            ),
            'platelet': MedicalTerm(
                term='Platelet Count',
                simple_name='Blood clotting cells',
                category='test',
                description='Platelets help your blood clot to stop bleeding. Both high and low counts can cause problems.',
                normal_range='150,000-400,000/μL',
                unit='cells/μL',
                high_meaning='May cause unwanted blood clots. Can be due to infection, inflammation, or bone marrow problems.',
                low_meaning='Increases bleeding risk. You may bruise easily or have prolonged bleeding from cuts.',
                related_conditions=['Thrombocytopenia', 'Thrombocytosis']
            ),
            
            # Kidney Function
            'creatinine': MedicalTerm(
                term='Serum Creatinine',
                simple_name='Kidney function marker',
                category='test',
                description='A waste product from muscle activity that healthy kidneys filter out. High levels suggest kidney problems.',
                normal_range='0.7-1.3 mg/dL',
                unit='mg/dL',
                high_meaning='Your kidneys may not be filtering waste properly. Can be due to kidney disease, dehydration, or certain medications.',
                low_meaning='Usually not concerning. May indicate low muscle mass.',
                related_conditions=['Chronic Kidney Disease', 'Acute Kidney Injury', 'Dehydration']
            ),
            'bun': MedicalTerm(
                term='Blood Urea Nitrogen',
                simple_name='Kidney waste marker',
                category='test',
                description='Measures nitrogen in your blood from urea, a waste product. Helps assess kidney function.',
                normal_range='7-20 mg/dL',
                unit='mg/dL',
                high_meaning='May indicate kidney problems, dehydration, high protein diet, or heart failure.',
                low_meaning='May indicate liver problems, malnutrition, or overhydration.',
                related_conditions=['Kidney Disease', 'Dehydration', 'Heart Failure']
            ),
            'uric_acid': MedicalTerm(
                term='Uric Acid',
                simple_name='Gout marker',
                category='test',
                description='A waste product from breaking down purines (found in some foods). High levels can cause gout.',
                normal_range='3.5-7.0 mg/dL',
                unit='mg/dL',
                high_meaning='Can cause gout (painful joint swelling) and kidney stones. May be due to diet, kidney problems, or genetics.',
                low_meaning='Usually not concerning.',
                related_conditions=['Gout', 'Kidney Stones']
            ),
            
            # Liver Function
            'sgpt': MedicalTerm(
                term='SGPT / ALT',
                simple_name='Liver enzyme',
                category='test',
                description='An enzyme found mainly in your liver. High levels suggest liver cell damage.',
                normal_range='7-56 U/L',
                unit='U/L',
                high_meaning='Indicates liver stress or damage. Common causes: fatty liver, hepatitis, alcohol, medications.',
                low_meaning='Usually normal.',
                related_conditions=['Fatty Liver', 'Hepatitis', 'Liver Damage']
            ),
            'sgot': MedicalTerm(
                term='SGOT / AST',
                simple_name='Liver/heart enzyme',
                category='test',
                description='An enzyme found in liver and heart. High levels may indicate damage to these organs.',
                normal_range='10-40 U/L',
                unit='U/L',
                high_meaning='May indicate liver damage, heart problems, or muscle injury.',
                low_meaning='Usually normal.',
                related_conditions=['Liver Disease', 'Heart Attack', 'Muscle Damage']
            ),
            'bilirubin': MedicalTerm(
                term='Bilirubin',
                simple_name='Jaundice marker',
                category='test',
                description='A yellow compound produced when red blood cells break down. High levels cause jaundice (yellow skin/eyes).',
                normal_range='0.1-1.2 mg/dL (total)',
                unit='mg/dL',
                high_meaning='Can cause yellowing of skin and eyes. May indicate liver problems, bile duct blockage, or rapid red blood cell breakdown.',
                low_meaning='Usually normal.',
                related_conditions=['Jaundice', 'Liver Disease', 'Gallstones']
            ),
            
            # Blood Sugar
            'fasting_glucose': MedicalTerm(
                term='Fasting Blood Sugar',
                simple_name='Blood sugar (empty stomach)',
                category='test',
                description='Measures blood sugar after not eating for 8+ hours. Used to diagnose diabetes.',
                normal_range='70-100 mg/dL',
                unit='mg/dL',
                high_meaning='100-125 mg/dL: Pre-diabetes. 126+ mg/dL: May indicate diabetes. Need to control diet and possibly take medication.',
                low_meaning='Below 70: Hypoglycemia. May cause shakiness, confusion, sweating. Need to eat something sugary.',
                related_conditions=['Diabetes', 'Pre-diabetes', 'Hypoglycemia']
            ),
            'hba1c': MedicalTerm(
                term='HbA1c',
                simple_name='3-month blood sugar average',
                category='test',
                description='Shows your average blood sugar over the past 2-3 months. Better indicator of diabetes control than single readings.',
                normal_range='Below 5.7%',
                unit='%',
                high_meaning='5.7-6.4%: Pre-diabetes. 6.5%+: Diabetes. Higher numbers mean blood sugar has been poorly controlled.',
                low_meaning='Very low may indicate frequent low blood sugar episodes.',
                related_conditions=['Diabetes', 'Pre-diabetes']
            ),
            
            # Lipid Profile
            'cholesterol': MedicalTerm(
                term='Total Cholesterol',
                simple_name='Total blood fat',
                category='test',
                description='Measures total cholesterol in blood. Too much can build up in arteries and cause heart problems.',
                normal_range='Below 200 mg/dL',
                unit='mg/dL',
                high_meaning='Increases heart disease risk. May need diet changes and possibly medication.',
                low_meaning='Very low cholesterol can also be concerning. Discuss with doctor.',
                related_conditions=['Heart Disease', 'Atherosclerosis', 'Stroke']
            ),
            'ldl': MedicalTerm(
                term='LDL Cholesterol',
                simple_name='Bad cholesterol',
                category='test',
                description='The "bad" cholesterol that can build up in artery walls and cause blockages.',
                normal_range='Below 100 mg/dL',
                unit='mg/dL',
                high_meaning='Increases risk of heart attack and stroke. Need to reduce through diet, exercise, and possibly medication.',
                low_meaning='Very low LDL is generally good.',
                related_conditions=['Heart Disease', 'Stroke']
            ),
            'hdl': MedicalTerm(
                term='HDL Cholesterol',
                simple_name='Good cholesterol',
                category='test',
                description='The "good" cholesterol that helps remove bad cholesterol from arteries.',
                normal_range='Above 40 mg/dL (men), Above 50 mg/dL (women)',
                unit='mg/dL',
                high_meaning='High HDL is protective against heart disease.',
                low_meaning='Low HDL increases heart disease risk. Can be improved with exercise and diet.',
                related_conditions=['Heart Disease']
            ),
            'triglycerides': MedicalTerm(
                term='Triglycerides',
                simple_name='Blood fats',
                category='test',
                description='A type of fat in blood. High levels increase heart disease risk.',
                normal_range='Below 150 mg/dL',
                unit='mg/dL',
                high_meaning='Increases heart disease risk. Often due to diet, obesity, or diabetes. Reduce sugary and fatty foods.',
                low_meaning='Very low is usually not concerning.',
                related_conditions=['Heart Disease', 'Pancreatitis']
            ),
            
            # Thyroid
            'tsh': MedicalTerm(
                term='TSH',
                simple_name='Thyroid control hormone',
                category='test',
                description='Controls thyroid hormone production. Abnormal levels indicate thyroid problems.',
                normal_range='0.4-4.0 mIU/L',
                unit='mIU/L',
                high_meaning='High TSH suggests underactive thyroid (hypothyroidism). Symptoms: fatigue, weight gain, feeling cold.',
                low_meaning='Low TSH suggests overactive thyroid (hyperthyroidism). Symptoms: weight loss, anxiety, rapid heartbeat.',
                related_conditions=['Hypothyroidism', 'Hyperthyroidism', 'Thyroid Nodules']
            ),
            't3': MedicalTerm(
                term='T3',
                simple_name='Thyroid hormone',
                category='test',
                description='One of the main thyroid hormones that controls metabolism.',
                normal_range='80-200 ng/dL',
                unit='ng/dL',
                high_meaning='May indicate overactive thyroid.',
                low_meaning='May indicate underactive thyroid.',
                related_conditions=['Thyroid Disorders']
            ),
            't4': MedicalTerm(
                term='T4',
                simple_name='Thyroid hormone',
                category='test',
                description='The main thyroid hormone. Converted to T3 in the body.',
                normal_range='5.0-12.0 μg/dL',
                unit='μg/dL',
                high_meaning='May indicate overactive thyroid.',
                low_meaning='May indicate underactive thyroid.',
                related_conditions=['Thyroid Disorders']
            ),
        }
        
        # Common Conditions Dictionary
        self.conditions: Dict[str, MedicalTerm] = {
            'diabetes': MedicalTerm(
                term='Diabetes Mellitus',
                simple_name='High blood sugar disease',
                category='condition',
                description='A condition where your body cannot properly use or produce insulin, leading to high blood sugar.',
                symptoms=['Frequent urination', 'Excessive thirst', 'Unexplained weight loss', 'Fatigue', 'Blurred vision', 'Slow healing wounds'],
                related_conditions=['Heart Disease', 'Kidney Disease', 'Neuropathy', 'Retinopathy']
            ),
            'hypertension': MedicalTerm(
                term='Hypertension',
                simple_name='High blood pressure',
                category='condition',
                description='Blood pressure consistently above 130/80 mmHg. Often has no symptoms but increases risk of heart attack and stroke.',
                symptoms=['Often none (silent killer)', 'Headaches', 'Shortness of breath', 'Nosebleeds (severe cases)'],
                related_conditions=['Heart Disease', 'Stroke', 'Kidney Disease']
            ),
            'anemia': MedicalTerm(
                term='Anemia',
                simple_name='Low blood count',
                category='condition',
                description='Not enough healthy red blood cells to carry oxygen to your body tissues.',
                symptoms=['Fatigue', 'Weakness', 'Pale skin', 'Shortness of breath', 'Dizziness', 'Cold hands and feet'],
                related_conditions=['Iron Deficiency', 'Vitamin B12 Deficiency', 'Chronic Disease']
            ),
            'hypothyroidism': MedicalTerm(
                term='Hypothyroidism',
                simple_name='Underactive thyroid',
                category='condition',
                description='Your thyroid gland does not produce enough thyroid hormone, slowing your metabolism.',
                symptoms=['Fatigue', 'Weight gain', 'Feeling cold', 'Dry skin', 'Depression', 'Constipation', 'Muscle weakness'],
                related_conditions=['Goiter', 'Heart Problems', 'Mental Health Issues']
            ),
            'hyperthyroidism': MedicalTerm(
                term='Hyperthyroidism',
                simple_name='Overactive thyroid',
                category='condition',
                description='Your thyroid gland produces too much thyroid hormone, speeding up your metabolism.',
                symptoms=['Weight loss', 'Rapid heartbeat', 'Anxiety', 'Tremors', 'Sweating', 'Difficulty sleeping'],
                related_conditions=['Graves Disease', 'Heart Problems', 'Osteoporosis']
            ),
            'fatty_liver': MedicalTerm(
                term='Fatty Liver Disease',
                simple_name='Fat buildup in liver',
                category='condition',
                description='Too much fat stored in liver cells. Often related to obesity, diabetes, or alcohol use.',
                symptoms=['Often none initially', 'Fatigue', 'Pain in upper right abdomen', 'Enlarged liver'],
                related_conditions=['Liver Cirrhosis', 'Diabetes', 'Heart Disease']
            ),
        }
        
        # Test name aliases for lookup
        self.test_aliases = {
            'hb': 'hemoglobin',
            'hgb': 'hemoglobin',
            'haemoglobin': 'hemoglobin',
            'red blood cell': 'rbc',
            'red blood cells': 'rbc',
            'white blood cell': 'wbc',
            'white blood cells': 'wbc',
            'tlc': 'wbc',
            'platelets': 'platelet',
            'plt': 'platelet',
            'serum creatinine': 'creatinine',
            's creatinine': 'creatinine',
            'blood urea': 'bun',
            'urea': 'bun',
            'alt': 'sgpt',
            'alanine transaminase': 'sgpt',
            'ast': 'sgot',
            'aspartate transaminase': 'sgot',
            'total bilirubin': 'bilirubin',
            'fbs': 'fasting_glucose',
            'fasting blood sugar': 'fasting_glucose',
            'fasting glucose': 'fasting_glucose',
            'glycated hemoglobin': 'hba1c',
            'glycosylated hemoglobin': 'hba1c',
            'total cholesterol': 'cholesterol',
            'ldl cholesterol': 'ldl',
            'hdl cholesterol': 'hdl',
            'tg': 'triglycerides',
            'thyroid stimulating hormone': 'tsh',
        }
    
    def lookup_test(self, test_name: str) -> Optional[MedicalTerm]:
        """
        Look up a medical test by name.
        
        Args:
            test_name: Test name (case insensitive)
            
        Returns:
            MedicalTerm if found, None otherwise
        """
        name_lower = test_name.lower().strip()
        
        # Check direct match
        if name_lower in self.lab_tests:
            return self.lab_tests[name_lower]
        
        # Check aliases
        if name_lower in self.test_aliases:
            return self.lab_tests.get(self.test_aliases[name_lower])
        
        # Fuzzy match - check if any test name is contained
        for key, term in self.lab_tests.items():
            if key in name_lower or name_lower in key:
                return term
            if term.term.lower() in name_lower or name_lower in term.term.lower():
                return term
        
        return None
    
    def lookup_condition(self, condition_name: str) -> Optional[MedicalTerm]:
        """
        Look up a medical condition by name.
        
        Args:
            condition_name: Condition name (case insensitive)
            
        Returns:
            MedicalTerm if found, None otherwise
        """
        name_lower = condition_name.lower().strip()
        
        # Check direct match
        if name_lower in self.conditions:
            return self.conditions[name_lower]
        
        # Fuzzy match
        for key, term in self.conditions.items():
            if key in name_lower or name_lower in key:
                return term
            if term.term.lower() in name_lower or name_lower in term.term.lower():
                return term
        
        return None
    
    def explain_test_result(
        self, 
        test_name: str, 
        value: float, 
        unit: str = ""
    ) -> Dict:
        """
        Explain a test result in simple terms.
        
        Args:
            test_name: Name of the test
            value: Test value
            unit: Unit of measurement
            
        Returns:
            Dictionary with explanation
        """
        term = self.lookup_test(test_name)
        
        if not term:
            return {
                'test': test_name,
                'value': value,
                'unit': unit,
                'found': False,
                'explanation': f"Test '{test_name}' not found in dictionary. Please consult your doctor for interpretation."
            }
        
        # Determine status
        status = 'unknown'
        meaning = ''
        
        if term.normal_range:
            # Try to parse normal range
            range_match = re.search(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', term.normal_range)
            if range_match:
                low = float(range_match.group(1))
                high = float(range_match.group(2))
                
                if value < low:
                    status = 'low'
                    meaning = term.low_meaning or "Below normal range."
                elif value > high:
                    status = 'high'
                    meaning = term.high_meaning or "Above normal range."
                else:
                    status = 'normal'
                    meaning = "Within normal range. This is good!"
        
        return {
            'test': term.term,
            'simple_name': term.simple_name,
            'value': value,
            'unit': unit or term.unit,
            'normal_range': term.normal_range,
            'status': status,
            'found': True,
            'description': term.description,
            'meaning': meaning,
            'related_conditions': term.related_conditions or []
        }
    
    def get_condition_info(self, condition_name: str) -> Dict:
        """
        Get information about a medical condition.
        
        Args:
            condition_name: Name of the condition
            
        Returns:
            Dictionary with condition information
        """
        term = self.lookup_condition(condition_name)
        
        if not term:
            return {
                'condition': condition_name,
                'found': False,
                'explanation': f"Condition '{condition_name}' not found. Please consult your doctor."
            }
        
        return {
            'condition': term.term,
            'simple_name': term.simple_name,
            'found': True,
            'description': term.description,
            'symptoms': term.symptoms or [],
            'related_conditions': term.related_conditions or []
        }
    
    def interpret_report(self, parsed_results: List[Dict]) -> List[Dict]:
        """
        Interpret a list of parsed medical results.
        
        Args:
            parsed_results: List of parsed test results from MedicalReportParser
            
        Returns:
            List of interpreted results with explanations
        """
        interpreted = []
        
        for result in parsed_results:
            test_name = result.get('test_name') or result.get('test_name_standardized', '')
            value = result.get('value')
            unit = result.get('unit', '')
            
            if test_name and value is not None:
                explanation = self.explain_test_result(test_name, float(value), unit)
                explanation['original_result'] = result
                interpreted.append(explanation)
            else:
                interpreted.append({
                    'original_result': result,
                    'found': False,
                    'explanation': 'Could not interpret this result'
                })
        
        return interpreted
    
    def get_all_tests(self) -> List[str]:
        """Get list of all known test names"""
        return list(self.lab_tests.keys())
    
    def get_all_conditions(self) -> List[str]:
        """Get list of all known conditions"""
        return list(self.conditions.keys())


# Convenience function
def explain_medical_term(term: str) -> str:
    """
    Quick function to explain a medical term.
    
    Args:
        term: Medical term to explain
        
    Returns:
        Simple explanation string
    """
    dictionary = MedicalDictionary(use_llm=False)
    
    # Try as test first
    test_info = dictionary.lookup_test(term)
    if test_info:
        return f"{test_info.term} ({test_info.simple_name}): {test_info.description}"
    
    # Try as condition
    condition_info = dictionary.lookup_condition(term)
    if condition_info:
        return f"{condition_info.term} ({condition_info.simple_name}): {condition_info.description}"
    
    return f"'{term}' not found in medical dictionary."


if __name__ == "__main__":
    # Test the medical dictionary
    print("=" * 60)
    print("  MEDICAL DICTIONARY TEST")
    print("=" * 60)
    
    dictionary = MedicalDictionary()
    
    # Test lookup
    print("\n1. Test Lookup:")
    tests = ['hemoglobin', 'HbA1c', 'creatinine', 'SGPT', 'TSH']
    for test in tests:
        info = dictionary.lookup_test(test)
        if info:
            print(f"   {test}: {info.simple_name}")
        else:
            print(f"   {test}: Not found")
    
    # Test result interpretation
    print("\n2. Result Interpretation:")
    result = dictionary.explain_test_result('hemoglobin', 8.5, 'g/dL')
    print(f"   Hemoglobin 8.5 g/dL: {result['status'].upper()}")
    print(f"   Meaning: {result['meaning']}")
    
    result2 = dictionary.explain_test_result('fasting glucose', 250, 'mg/dL')
    print(f"\n   Fasting Glucose 250 mg/dL: {result2['status'].upper()}")
    print(f"   Meaning: {result2['meaning']}")
    
    # Condition lookup
    print("\n3. Condition Lookup:")
    conditions = ['diabetes', 'anemia', 'hypothyroidism']
    for cond in conditions:
        info = dictionary.get_condition_info(cond)
        if info['found']:
            print(f"   {info['condition']}: {info['simple_name']}")
            if info['symptoms']:
                print(f"      Symptoms: {', '.join(info['symptoms'][:3])}...")
    
    print("\n" + "=" * 60)
    print("  TEST COMPLETE")
    print("=" * 60)
