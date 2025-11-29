"""Analyze rental agreement document"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.plainsense_api import PlainSenseAPI

api = PlainSenseAPI(domain='legal')
result = api.process_legal_document(r'D:\PlainSense\PlainSense\Dataset\Rental Dataset\rental_agreement_002.docx')

print(f"\n{'='*70}")
print(f"RENTAL AGREEMENT ANALYSIS")
print(f"{'='*70}")
print(f"Total Clauses: {len(result.clauses)}")
print(f"{'='*70}\n")

for i, c in enumerate(result.clauses):
    original = c.get('original', '')
    english = c.get('english', {})
    plain = english.get('plain', '') if isinstance(english, dict) else ''
    risk = c.get('risk', {})
    risk_level = risk.get('level', 'unknown') if isinstance(risk, dict) else 'unknown'
    
    print(f"{'─'*70}")
    print(f"CLAUSE {i+1} | Risk: {risk_level.upper()}")
    print(f"{'─'*70}")
    
    print(f"\n📜 ORIGINAL:")
    print(f"   {original}")
    
    print(f"\n✅ SIMPLIFIED:")
    print(f"   {plain}")
    
    print()

# Show law references
if result.law_references:
    print(f"\n{'='*70}")
    print(f"📚 LAW REFERENCES DETECTED (for popup explanations)")
    print(f"{'='*70}")
    for i, law in enumerate(result.law_references, 1):
        print(f"\n{i}. {law['law_name']}")
        print(f"   Reference: {law['full_reference']}")
        if law.get('section'):
            print(f"   Section: {law['section']}")
        print(f"   Explanation: {law['explanation'][:150]}...")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}")
