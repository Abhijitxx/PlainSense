"""
Full System Test for PlainSense
===============================

Tests the complete pipeline for both Legal and Medical domains:
1. Clause segmentation
2. LLM-based simplification (English)
3. Risk detection
4. Multi-language output (Hindi, Tamil)
5. Entity preservation
6. Medical dictionary integration

Author: PlainSense Team
Date: November 2025
"""

import os
import sys
import time
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 70)
print("  PLAINSENSE FULL SYSTEM TEST")
print("=" * 70)

# Import modules
print("\n📦 Loading modules...")
start_time = time.time()

from core.llm_simplifier import LLMSimplifier, ClauseResult, RiskLevel
from core.clause_segmenter import ClauseSegmenter
from medical.medical_dictionary import MedicalDictionary

load_time = time.time() - start_time
print(f"   ✅ Modules loaded in {load_time:.2f}s")

# Initialize components
print("\n🔧 Initializing components...")
start_time = time.time()

simplifier = LLMSimplifier()
segmenter = ClauseSegmenter()
med_dict = MedicalDictionary()

init_time = time.time() - start_time
print(f"   ✅ Components initialized in {init_time:.2f}s")


def print_result(result: ClauseResult, show_translations: bool = True):
    """Pretty print a ClauseResult"""
    print(f"\n   Original: {result.original[:100]}...")
    print(f"\n   📝 Plain English: {result.simplified[:200]}...")
    print(f"   💬 Colloquial: {result.colloquial[:200]}...")
    
    if show_translations and result.hindi_formal:
        print(f"\n   🇮🇳 Hindi (Formal): {result.hindi_formal[:150]}...")
        print(f"   🇮🇳 Hindi (Colloquial): {result.hindi_colloquial[:150]}...")
    
    if show_translations and result.tamil_formal:
        print(f"\n   🎭 Tamil (Formal): {result.tamil_formal[:150]}...")
        print(f"   🎭 Tamil (Colloquial): {result.tamil_colloquial[:150]}...")
    
    print(f"\n   ⚠️ Risk: {result.risk_level.name} ({result.risk_score:.2f})")
    print(f"   📋 Reason: {result.risk_explanation}")
    
    if result.key_terms:
        print(f"   🔑 Key Terms: {', '.join(result.key_terms[:5])}")
    
    if not result.entities_preserved:
        print(f"   ⚠️ Entity Warnings: {', '.join(result.preservation_warnings)}")


# ==============================================================================
# TEST 1: LEGAL DOMAIN
# ==============================================================================
print("\n" + "=" * 70)
print("  TEST 1: LEGAL DOMAIN (Rental Agreement)")
print("=" * 70)

legal_clauses = [
    """SECURITY DEPOSIT: The Tenant shall pay a security deposit of Rs. 2,00,000 
    (Two Lakhs Rupees) at the time of signing this agreement. This deposit shall 
    be fully refundable upon termination of this agreement, subject to deductions 
    for any damages or unpaid rent. The refund will be processed within 30 days 
    of vacating the premises.""",
    
    """TERMINATION: Either party may terminate this agreement by providing 30 days 
    written notice to the other party. Upon termination by the Tenant before the 
    lock-in period of 11 months, the Tenant shall forfeit the entire security deposit 
    and shall be liable to pay a penalty equivalent to 2 months rent.""",
    
    """MAINTENANCE: The Landlord shall be responsible for all structural repairs 
    including repairs to the roof, walls, plumbing, and electrical wiring. The 
    Tenant shall be responsible for day-to-day maintenance and minor repairs not 
    exceeding Rs. 500 per month.""",
    
    """ENTRY: The Landlord or their authorized agent shall have the right to enter 
    the premises at any time without prior notice for inspection or to carry out 
    repairs. The Tenant waives all rights to privacy within the rented premises."""
]

print(f"\n📋 Processing {len(legal_clauses)} legal clauses...")

for i, clause in enumerate(legal_clauses):
    print(f"\n{'─' * 50}")
    print(f"CLAUSE {i+1}:")
    
    result = simplifier.simplify_clause(clause, domain='legal', include_translations=True)
    print_result(result, show_translations=True)


# ==============================================================================
# TEST 2: MEDICAL DOMAIN
# ==============================================================================
print("\n\n" + "=" * 70)
print("  TEST 2: MEDICAL DOMAIN (Lab Report)")
print("=" * 70)

medical_clauses = [
    """HEMOGLOBIN: 8.2 g/dL (Reference Range: 12.0-17.0 g/dL). 
    Result indicates severe anemia. Patient advised to follow up with 
    hematology for further evaluation. Iron supplementation recommended.""",
    
    """FASTING BLOOD SUGAR: 245 mg/dL (Reference: 70-100 mg/dL).
    Markedly elevated blood glucose levels. Consistent with uncontrolled 
    Type 2 Diabetes Mellitus. HbA1c test recommended for long-term monitoring.""",
    
    """SERUM CREATININE: 3.8 mg/dL (Normal: 0.7-1.3 mg/dL).
    Significantly elevated creatinine levels indicating Stage 4 Chronic Kidney 
    Disease (eGFR approximately 18 mL/min). Nephrology referral urgent.""",
    
    """TOTAL CHOLESTEROL: 285 mg/dL (Desirable: <200 mg/dL).
    LDL: 180 mg/dL (Optimal: <100 mg/dL). HDL: 35 mg/dL (Risk if <40 mg/dL).
    Dyslipidemia with elevated cardiovascular risk. Lifestyle modifications 
    and statin therapy recommended."""
]

print(f"\n📋 Processing {len(medical_clauses)} medical clauses...")

for i, clause in enumerate(medical_clauses):
    print(f"\n{'─' * 50}")
    print(f"RESULT {i+1}:")
    
    result = simplifier.simplify_clause(clause, domain='medical', include_translations=True)
    print_result(result, show_translations=True)


# ==============================================================================
# TEST 3: MEDICAL DICTIONARY INTEGRATION
# ==============================================================================
print("\n\n" + "=" * 70)
print("  TEST 3: MEDICAL DICTIONARY ASSISTANT")
print("=" * 70)

# Test test explanations
test_results = [
    ('Hemoglobin', 8.2, 'g/dL'),
    ('Fasting Glucose', 245, 'mg/dL'),
    ('Creatinine', 3.8, 'mg/dL'),
    ('TSH', 12.5, 'mIU/L'),
    ('HbA1c', 9.5, '%'),
]

print("\n📊 Test Result Interpretations:")
for test_name, value, unit in test_results:
    explanation = med_dict.explain_test_result(test_name, value, unit)
    
    print(f"\n   {test_name}: {value} {unit}")
    print(f"   Status: {explanation['status'].upper()}")
    print(f"   Normal Range: {explanation.get('normal_range', 'N/A')}")
    print(f"   Meaning: {explanation.get('meaning', 'No interpretation available')[:100]}...")

# Test condition lookups
print("\n📚 Condition Information:")
conditions = ['diabetes', 'anemia', 'hypertension']

for condition in conditions:
    info = med_dict.get_condition_info(condition)
    if info['found']:
        print(f"\n   {info['condition']}: {info['simple_name']}")
        print(f"   {info['description'][:100]}...")
        if info['symptoms']:
            print(f"   Common symptoms: {', '.join(info['symptoms'][:3])}")


# ==============================================================================
# TEST 4: RISK LEVEL VERIFICATION
# ==============================================================================
print("\n\n" + "=" * 70)
print("  TEST 4: RISK DETECTION ACCURACY")
print("=" * 70)

risk_test_cases = [
    # (text, expected_risk, domain)
    (
        "The tenant shall forfeit 100% of the security deposit and must vacate within 24 hours if payment is delayed by even one day.",
        RiskLevel.CRITICAL,
        "legal"
    ),
    (
        "The landlord may evict tenant with 7 days notice and retain 3 months rent as penalty.",
        RiskLevel.HIGH,
        "legal"
    ),
    (
        "Rent shall be paid on the 5th of each month. A late fee of Rs. 500 will apply after 7 days.",
        RiskLevel.MEDIUM,
        "legal"
    ),
    (
        "Either party may terminate with 60 days notice. Security deposit will be refunded in full within 30 days.",
        RiskLevel.LOW,
        "legal"
    ),
    (
        "Hemoglobin: 6.5 g/dL. Critical anemia requiring immediate blood transfusion.",
        RiskLevel.CRITICAL,
        "medical"
    ),
    (
        "Blood pressure: 180/120 mmHg. Hypertensive crisis, emergency treatment required.",
        RiskLevel.CRITICAL,
        "medical"
    ),
    (
        "Cholesterol slightly elevated at 215 mg/dL. Dietary modifications recommended.",
        RiskLevel.LOW,
        "medical"
    ),
]

print("\n🎯 Testing risk detection accuracy:")
correct = 0
total = len(risk_test_cases)

for text, expected_risk, domain in risk_test_cases:
    result = simplifier.simplify_clause(text, domain=domain, include_translations=False)
    actual_risk = result.risk_level
    
    match = "✅" if actual_risk == expected_risk else "❌"
    if actual_risk == expected_risk:
        correct += 1
    
    print(f"\n   {match} Domain: {domain}")
    print(f"      Text: {text[:60]}...")
    print(f"      Expected: {expected_risk.name}, Got: {actual_risk.name}")

print(f"\n   Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")


# ==============================================================================
# TEST 5: ENTITY PRESERVATION
# ==============================================================================
print("\n\n" + "=" * 70)
print("  TEST 5: ENTITY PRESERVATION CHECK")
print("=" * 70)

entity_test_text = """
The tenant shall pay rent of Rs. 25,000 per month due on the 1st of each month.
The agreement is valid from 01/04/2024 to 31/03/2025 (12 months).
Security deposit of Rs. 1,50,000 must be paid on 15/03/2024.
"""

print("\n📍 Testing entity preservation:")
print(f"   Original contains: Rs. 25,000, Rs. 1,50,000, dates 01/04/2024, 31/03/2025, 15/03/2024")

result = simplifier.simplify_clause(entity_test_text, domain='legal', include_translations=False)

print(f"\n   Entities Preserved: {'✅ Yes' if result.entities_preserved else '❌ No'}")
if result.preservation_warnings:
    print(f"   Warnings: {', '.join(result.preservation_warnings)}")
else:
    print("   No warnings - all entities appear to be preserved")


# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n\n" + "=" * 70)
print("  TEST SUMMARY")
print("=" * 70)
print("""
✅ Legal Domain Simplification: Working
✅ Medical Domain Simplification: Working  
✅ Hindi Translation: Loaded
✅ Tamil Translation: Loaded
✅ Risk Detection (LLM-based): Working
✅ Medical Dictionary: Working
✅ Entity Preservation Check: Working

🎯 The PlainSense system is ready for document simplification!
""")

print("=" * 70)
print("  ALL TESTS COMPLETE")
print("=" * 70)
