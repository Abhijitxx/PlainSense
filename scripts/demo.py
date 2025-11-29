"""
PlainSense Demo Script
======================

Demonstrates the full PlainSense document simplification system.
Shows both Legal and Medical document processing with all features.

Usage:
    python demo.py

Author: PlainSense Team
Date: November 2025
"""

import os
import sys
import time
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.plainsense_api import PlainSenseAPI


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_clause_result(clause: dict, index: int):
    """Print a formatted clause result"""
    print(f"\n{'─' * 50}")
    print(f"📋 CLAUSE {index}")
    print(f"{'─' * 50}")
    
    # Original
    original = clause.get('original', 'N/A')
    print(f"\n📄 Original ({len(original)} chars):")
    print(f"   {original[:200]}{'...' if len(original) > 200 else ''}")
    
    # English versions
    english = clause.get('english', {})
    print(f"\n✅ Plain English:")
    print(f"   {english.get('plain', 'N/A')[:200]}...")
    
    print(f"\n💬 Colloquial English:")
    print(f"   {english.get('colloquial', 'N/A')[:200]}...")
    
    # Hindi versions
    hindi = clause.get('hindi', {})
    if hindi.get('formal'):
        print(f"\n🇮🇳 Hindi (Formal):")
        print(f"   {hindi.get('formal', 'N/A')[:150]}...")
    
    # Tamil versions
    tamil = clause.get('tamil', {})
    if tamil.get('formal'):
        print(f"\n🎭 Tamil (Formal):")
        print(f"   {tamil.get('formal', 'N/A')[:150]}...")
    
    # Risk assessment
    risk = clause.get('risk', {})
    risk_level = risk.get('level', 'UNKNOWN')
    risk_score = risk.get('score', 0)
    
    emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢', 'NONE': '⚪'}.get(risk_level, '❓')
    print(f"\n{emoji} Risk Level: {risk_level} (Score: {risk_score:.2f})")
    print(f"   Explanation: {risk.get('explanation', 'N/A')}")
    
    # Key terms
    key_terms = clause.get('key_terms', [])
    if key_terms:
        print(f"\n🔑 Key Terms: {', '.join(key_terms[:5])}")
    
    # Entity preservation
    if not clause.get('entities_preserved', True):
        warnings = clause.get('preservation_warnings', [])
        print(f"\n⚠️ Entity Warnings: {', '.join(warnings)}")


def demo_legal_processing(api: PlainSenseAPI):
    """Demonstrate legal document processing"""
    print_header("LEGAL DOCUMENT PROCESSING DEMO")
    
    # Sample rental agreement clauses
    legal_text = """
    RENTAL AGREEMENT
    
    1. SECURITY DEPOSIT: The Tenant agrees to pay a security deposit of Rs. 2,00,000 
    (Two Lakhs Rupees) at the time of signing this agreement. This deposit shall be 
    fully refundable upon termination of this agreement, subject to deductions for 
    any damages or unpaid rent. The refund will be processed within 30 days of vacating.
    
    2. TERMINATION CLAUSE: Either party may terminate this agreement by providing 30 days 
    written notice to the other party. Upon termination by the Tenant before the lock-in 
    period of 11 months, the Tenant shall forfeit the entire security deposit and shall 
    be liable to pay a penalty equivalent to 2 months rent.
    
    3. LANDLORD'S RIGHT OF ENTRY: The Landlord or their authorized agent shall have the 
    right to enter the premises at any time without prior notice for inspection or to 
    carry out repairs. The Tenant waives all rights to privacy within the rented premises.
    
    4. MAINTENANCE: The Landlord shall be responsible for all structural repairs including 
    repairs to the roof, walls, plumbing, and electrical wiring. The Tenant shall be 
    responsible for day-to-day maintenance and minor repairs not exceeding Rs. 500 per month.
    """
    
    print("\n📜 Processing Rental Agreement...")
    print(f"   Input: {len(legal_text)} characters")
    
    # Process with translations disabled for faster demo
    result = api.process_legal_document(legal_text, include_translations=False)
    
    print(f"\n⏱️ Processing Time: {result.processing_time:.2f}s")
    print(f"📊 Clauses Found: {result.summary['total_clauses']}")
    
    # Print each clause
    for i, clause in enumerate(result.clauses, 1):
        print_clause_result(clause, i)
    
    # Print summary
    print_header("LEGAL DOCUMENT SUMMARY")
    summary = result.summary
    
    print(f"\n📈 Risk Breakdown:")
    for level, count in summary['risk_breakdown'].items():
        if count > 0:
            emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢', 'NONE': '⚪'}.get(level, '❓')
            print(f"   {emoji} {level}: {count} clauses")
    
    print(f"\n⚠️ Overall Risk: {summary['overall_risk']}")
    
    print(f"\n💡 Recommendations:")
    for rec in summary.get('recommendations', []):
        print(f"   • {rec}")


def demo_medical_processing(api: PlainSenseAPI):
    """Demonstrate medical document processing"""
    print_header("MEDICAL DOCUMENT PROCESSING DEMO")
    
    # Sample medical report
    medical_text = """
    COMPLETE BLOOD COUNT (CBC) REPORT
    
    Patient: John Doe
    Date: 28-Nov-2025
    
    HEMOGLOBIN: 8.5 g/dL (Reference Range: 12.0-17.0 g/dL)
    Result indicates severe anemia. Patient advised to follow up with 
    hematology for further evaluation. Iron supplementation recommended.
    
    WHITE BLOOD CELL COUNT: 12,500 /μL (Reference: 4,000-11,000 /μL)
    Slightly elevated WBC count. May indicate mild infection or inflammation.
    Clinical correlation recommended.
    
    FASTING BLOOD SUGAR: 245 mg/dL (Reference: 70-100 mg/dL)
    Markedly elevated blood glucose levels. Consistent with uncontrolled 
    Type 2 Diabetes Mellitus. HbA1c test recommended for long-term monitoring.
    
    SERUM CREATININE: 1.8 mg/dL (Normal: 0.7-1.3 mg/dL)
    Moderately elevated creatinine levels indicating Stage 3 Chronic Kidney 
    Disease (eGFR approximately 42 mL/min). Nephrology referral recommended.
    """
    
    print("\n🏥 Processing Medical Report...")
    print(f"   Input: {len(medical_text)} characters")
    
    result = api.process_medical_document(medical_text, include_translations=False)
    
    print(f"\n⏱️ Processing Time: {result.processing_time:.2f}s")
    print(f"📊 Sections Found: {result.summary['total_sections']}")
    
    # Print each section
    for i, section in enumerate(result.clauses, 1):
        print_clause_result(section, i)
    
    # Print summary
    print_header("MEDICAL REPORT SUMMARY")
    summary = result.summary
    
    print(f"\n📈 Risk Breakdown:")
    for level, count in summary['risk_breakdown'].items():
        if count > 0:
            emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢', 'NONE': '⚪'}.get(level, '❓')
            print(f"   {emoji} {level}: {count} results")
    
    print(f"\n⚠️ Overall Health Risk: {summary['overall_health_risk']}")
    
    print(f"\n💡 Recommendations:")
    for rec in summary.get('recommendations', []):
        print(f"   • {rec}")


def demo_medical_dictionary(api: PlainSenseAPI):
    """Demonstrate medical dictionary feature"""
    print_header("MEDICAL DICTIONARY ASSISTANT DEMO")
    
    # Test lookups
    test_terms = ['hemoglobin', 'creatinine', 'TSH', 'diabetes', 'anemia']
    
    print("\n📚 Medical Term Explanations:\n")
    
    for term in test_terms:
        info = api.explain_medical_term(term)
        
        if info.get('found'):
            print(f"✅ {info.get('term', term)}:")
            print(f"   Simple Name: {info.get('simple_name', 'N/A')}")
            print(f"   Description: {info.get('description', 'N/A')[:100]}...")
            if info.get('normal_range'):
                print(f"   Normal Range: {info.get('normal_range')}")
            print()
        else:
            print(f"❌ {term}: Not found in dictionary\n")
    
    # Test result interpretation
    print("\n🔬 Lab Result Interpretations:\n")
    
    test_results = [
        ('Hemoglobin', 8.5, 'g/dL'),
        ('Fasting Glucose', 245, 'mg/dL'),
        ('Creatinine', 1.8, 'mg/dL'),
        ('TSH', 12.5, 'mIU/L'),
    ]
    
    for test_name, value, unit in test_results:
        interpretation = api.interpret_lab_result(test_name, value, unit)
        
        status = interpretation.get('status', 'unknown').upper()
        emoji = {'HIGH': '🔴', 'LOW': '🔵', 'NORMAL': '🟢', 'UNKNOWN': '⚪'}.get(status, '❓')
        
        print(f"{emoji} {test_name}: {value} {unit}")
        print(f"   Status: {status}")
        print(f"   Normal Range: {interpretation.get('normal_range', 'N/A')}")
        print(f"   Meaning: {interpretation.get('meaning', 'N/A')[:100]}...")
        print()


def main():
    """Main demo function"""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  PLAINSENSE - Document Simplification System".center(68) + "█")
    print("█" + "  Final Year Project Demo".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Initialize API
    print("\n🚀 Initializing PlainSense API...")
    start = time.time()
    api = PlainSenseAPI(enable_translations=False)  # Disable translations for faster demo
    print(f"✅ API initialized in {time.time() - start:.1f}s")
    
    # Run demos
    try:
        demo_legal_processing(api)
        demo_medical_processing(api)
        demo_medical_dictionary(api)
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during demo: {e}")
        raise
    
    # Summary
    print_header("DEMO COMPLETE")
    print("""
✅ PlainSense System Features Demonstrated:

   📜 LEGAL DOCUMENTS:
      • Clause segmentation
      • Plain English simplification
      • Colloquial/friendly simplification
      • Risk detection (CRITICAL/HIGH/MEDIUM/LOW)
      • Key term extraction
      • Entity preservation checking

   🏥 MEDICAL DOCUMENTS:
      • Lab report parsing
      • Medical term simplification
      • Risk assessment for abnormal values
      • Medical dictionary integration

   🌐 MULTI-LANGUAGE (when enabled):
      • Hindi formal translation
      • Hindi colloquial translation
      • Tamil formal translation
      • Tamil colloquial translation

Thank you for using PlainSense! 🎉
""")


if __name__ == "__main__":
    main()
