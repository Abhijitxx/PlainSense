"""
Domain Classification Module
============================

Classifies documents as:
- LEGAL (rental agreements, contracts)
- MEDICAL (lab reports, diagnostic reports)

Uses lightweight keyword-based classification with optional
transformer-based classification for higher accuracy.

Author: PlainSense Team
Date: November 2025
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from collections import Counter


class Domain(Enum):
    """Document domain types"""
    LEGAL = "legal"
    MEDICAL = "medical"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Classification result with confidence"""
    domain: Domain
    confidence: float
    legal_score: float
    medical_score: float
    top_indicators: List[str]
    method: str  # 'keyword', 'transformer', 'hybrid'
    
    def to_dict(self) -> Dict:
        return {
            'domain': self.domain.value,
            'confidence': round(self.confidence, 3),
            'legal_score': round(self.legal_score, 3),
            'medical_score': round(self.medical_score, 3),
            'top_indicators': self.top_indicators,
            'method': self.method
        }


class DomainClassifier:
    """
    Classifies documents into LEGAL or MEDICAL domain
    """
    
    def __init__(self, use_transformer: bool = False):
        """
        Initialize classifier
        
        Args:
            use_transformer: Whether to use transformer model (requires transformers library)
        """
        self.use_transformer = use_transformer
        self._transformer_model = None
        
        # Legal document keywords with weights
        self.legal_keywords = {
            # High-weight legal terms
            'agreement': 3.0,
            'contract': 3.0,
            'landlord': 3.0,
            'tenant': 3.0,
            'lessor': 3.0,
            'lessee': 3.0,
            'lease': 3.0,
            'rental': 2.5,
            'rent': 2.0,
            'premises': 2.5,
            'property': 1.5,
            
            # Legal structure terms
            'whereas': 2.5,
            'witnesseth': 3.0,
            'hereby': 2.0,
            'hereinafter': 2.5,
            'herein': 2.0,
            'thereof': 2.0,
            'therein': 2.0,
            'aforementioned': 2.0,
            
            # Contract terms
            'party': 1.5,
            'parties': 1.5,
            'executed': 1.5,
            'enforceable': 2.0,
            'binding': 2.0,
            'clause': 1.5,
            'article': 1.5,
            'section': 1.0,
            'schedule': 1.5,
            'annexure': 2.0,
            
            # Legal actions
            'termination': 2.0,
            'terminate': 2.0,
            'notice period': 2.5,
            'eviction': 2.5,
            'evict': 2.5,
            'breach': 2.0,
            'violation': 2.0,
            'penalty': 2.0,
            'penalties': 2.0,
            'damages': 2.0,
            'liable': 2.0,
            'liability': 2.0,
            'indemnify': 2.5,
            'indemnification': 2.5,
            
            # Financial legal terms
            'security deposit': 2.5,
            'caution deposit': 2.5,
            'advance rent': 2.0,
            'monthly rent': 2.0,
            'lock-in': 2.5,
            'lock in period': 2.5,
            
            # Signatures
            'signature': 1.5,
            'signed': 1.5,
            'witness': 1.5,
            'notarized': 2.0,
            'stamp duty': 2.0,
            'registration': 1.5,
            
            # Legal jurisdiction
            'jurisdiction': 2.0,
            'arbitration': 2.0,
            'dispute resolution': 2.0,
            'court': 1.5,
            'law': 1.0,
        }
        
        # Medical document keywords with weights
        self.medical_keywords = {
            # High-weight medical terms
            'patient': 2.5,
            'specimen': 2.5,
            'sample': 2.0,
            'laboratory': 2.5,
            'pathology': 3.0,
            'pathologist': 3.0,
            'diagnosis': 2.5,
            'diagnostic': 2.5,
            
            # Test-related
            'test': 1.5,
            'investigation': 2.0,
            'result': 1.5,
            'results': 1.5,
            'reference range': 3.0,
            'normal range': 3.0,
            'abnormal': 2.5,
            'normal': 1.0,
            
            # Blood tests
            'blood': 2.0,
            'hemoglobin': 3.0,
            'haemoglobin': 3.0,
            'rbc': 2.5,
            'wbc': 2.5,
            'platelet': 3.0,
            'hematocrit': 3.0,
            'haematocrit': 3.0,
            'cbc': 3.0,
            'complete blood count': 3.0,
            
            # Blood chemistry
            'glucose': 2.5,
            'blood sugar': 3.0,
            'fasting': 2.0,
            'creatinine': 3.0,
            'urea': 2.5,
            'cholesterol': 3.0,
            'triglycerides': 3.0,
            'bilirubin': 3.0,
            
            # Units
            'mg/dl': 3.0,
            'g/dl': 3.0,
            'gm/dl': 3.0,
            'cells/cumm': 3.0,
            '/cumm': 2.5,
            'million/cumm': 3.0,
            'iu/l': 2.5,
            'u/l': 2.0,
            'meq/l': 2.5,
            'mmol/l': 2.5,
            
            # Medical entities
            'doctor': 1.5,
            'hospital': 2.0,
            'clinic': 2.0,
            'lab': 1.5,
            'medical': 1.5,
            'clinical': 2.0,
            
            # Specific tests
            'thyroid': 2.5,
            'tsh': 3.0,
            'liver function': 3.0,
            'kidney function': 3.0,
            'lipid profile': 3.0,
            'urine': 2.0,
            'urinalysis': 3.0,
            
            # Flags
            'high': 1.0,
            'low': 1.0,
            'critical': 2.0,
            'elevated': 2.0,
            'decreased': 2.0,
        }
    
    def classify_keyword(self, text: str) -> ClassificationResult:
        """
        Classify using keyword matching
        
        Args:
            text: Document text
            
        Returns:
            ClassificationResult with scores and confidence
        """
        text_lower = text.lower()
        
        # Calculate legal score
        legal_score = 0.0
        legal_found = []
        for keyword, weight in self.legal_keywords.items():
            count = len(re.findall(rf'\b{re.escape(keyword)}\b', text_lower))
            if count > 0:
                legal_score += weight * min(count, 5)  # Cap at 5 occurrences
                legal_found.append((keyword, count * weight))
        
        # Calculate medical score
        medical_score = 0.0
        medical_found = []
        for keyword, weight in self.medical_keywords.items():
            # Handle units with special characters
            if '/' in keyword:
                count = text_lower.count(keyword)
            else:
                count = len(re.findall(rf'\b{re.escape(keyword)}\b', text_lower))
            
            if count > 0:
                medical_score += weight * min(count, 5)
                medical_found.append((keyword, count * weight))
        
        # Normalize scores
        total = legal_score + medical_score
        if total > 0:
            legal_norm = legal_score / total
            medical_norm = medical_score / total
        else:
            legal_norm = 0.5
            medical_norm = 0.5
        
        # Determine domain
        if medical_norm > legal_norm + 0.1:
            domain = Domain.MEDICAL
            confidence = medical_norm
        elif legal_norm > medical_norm + 0.1:
            domain = Domain.LEGAL
            confidence = legal_norm
        else:
            domain = Domain.UNKNOWN
            confidence = max(legal_norm, medical_norm)
        
        # Get top indicators
        if domain == Domain.MEDICAL:
            top_indicators = [k for k, _ in sorted(medical_found, key=lambda x: -x[1])[:5]]
        elif domain == Domain.LEGAL:
            top_indicators = [k for k, _ in sorted(legal_found, key=lambda x: -x[1])[:5]]
        else:
            all_found = legal_found + medical_found
            top_indicators = [k for k, _ in sorted(all_found, key=lambda x: -x[1])[:5]]
        
        return ClassificationResult(
            domain=domain,
            confidence=confidence,
            legal_score=legal_norm,
            medical_score=medical_norm,
            top_indicators=top_indicators,
            method='keyword'
        )
    
    def classify_transformer(self, text: str) -> ClassificationResult:
        """
        Classify using transformer model (if available)
        
        Args:
            text: Document text
            
        Returns:
            ClassificationResult
        """
        try:
            from transformers import pipeline
            
            if self._transformer_model is None:
                # Use zero-shot classification
                self._transformer_model = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
            
            # Truncate text for transformer
            text_truncated = text[:1000]
            
            result = self._transformer_model(
                text_truncated,
                candidate_labels=["legal document", "medical report"],
                hypothesis_template="This is a {}."
            )
            
            labels = result['labels']
            scores = result['scores']
            
            if labels[0] == "legal document":
                domain = Domain.LEGAL
                legal_score = scores[0]
                medical_score = scores[1]
            else:
                domain = Domain.MEDICAL
                medical_score = scores[0]
                legal_score = scores[1]
            
            return ClassificationResult(
                domain=domain,
                confidence=max(scores),
                legal_score=legal_score,
                medical_score=medical_score,
                top_indicators=[],
                method='transformer'
            )
            
        except ImportError:
            print("⚠️ Transformers library not available, falling back to keyword classification")
            return self.classify_keyword(text)
        except Exception as e:
            print(f"⚠️ Transformer classification failed: {e}, falling back to keyword")
            return self.classify_keyword(text)
    
    def classify(self, text: str, method: str = 'keyword') -> ClassificationResult:
        """
        Classify document domain
        
        Args:
            text: Document text
            method: 'keyword', 'transformer', or 'hybrid'
            
        Returns:
            ClassificationResult
        """
        if method == 'transformer' and self.use_transformer:
            return self.classify_transformer(text)
        elif method == 'hybrid' and self.use_transformer:
            # Combine both methods
            keyword_result = self.classify_keyword(text)
            transformer_result = self.classify_transformer(text)
            
            # Average the scores
            avg_legal = (keyword_result.legal_score + transformer_result.legal_score) / 2
            avg_medical = (keyword_result.medical_score + transformer_result.medical_score) / 2
            
            if avg_medical > avg_legal + 0.05:
                domain = Domain.MEDICAL
                confidence = avg_medical
            elif avg_legal > avg_medical + 0.05:
                domain = Domain.LEGAL
                confidence = avg_legal
            else:
                domain = Domain.UNKNOWN
                confidence = max(avg_legal, avg_medical)
            
            return ClassificationResult(
                domain=domain,
                confidence=confidence,
                legal_score=avg_legal,
                medical_score=avg_medical,
                top_indicators=keyword_result.top_indicators,
                method='hybrid'
            )
        else:
            return self.classify_keyword(text)
    
    def get_domain_tokens(self, text: str) -> str:
        """
        Get domain token for simplification model
        
        Args:
            text: Document text
            
        Returns:
            Domain token string: '[LEGAL]' or '[MEDICAL]'
        """
        result = self.classify(text)
        
        if result.domain == Domain.LEGAL:
            return '[LEGAL]'
        elif result.domain == Domain.MEDICAL:
            return '[MEDICAL]'
        else:
            # Default based on higher score
            if result.legal_score > result.medical_score:
                return '[LEGAL]'
            else:
                return '[MEDICAL]'


# =============================================================================
# Convenience Functions
# =============================================================================

def classify_document(text: str) -> Dict:
    """
    Quick function to classify document domain
    
    Args:
        text: Document text
        
    Returns:
        Classification result as dictionary
    """
    classifier = DomainClassifier()
    result = classifier.classify(text)
    return result.to_dict()


def get_domain(text: str) -> str:
    """
    Get domain as string
    
    Args:
        text: Document text
        
    Returns:
        'legal', 'medical', or 'unknown'
    """
    classifier = DomainClassifier()
    result = classifier.classify(text)
    return result.domain.value


def get_domain_token(text: str) -> str:
    """
    Get domain token for model input
    
    Args:
        text: Document text
        
    Returns:
        '[LEGAL]' or '[MEDICAL]'
    """
    classifier = DomainClassifier()
    return classifier.get_domain_tokens(text)


# =============================================================================
# Main - Demo
# =============================================================================

if __name__ == '__main__':
    # Test samples
    legal_sample = """
    RENTAL AGREEMENT
    
    This Agreement is made between the Landlord and the Tenant.
    The Landlord agrees to let the premises to the Tenant for monthly rent.
    The security deposit shall be refunded within 30 days.
    Notice period of 2 months is required for termination.
    Any breach may result in penalties and eviction.
    """
    
    medical_sample = """
    LABORATORY REPORT
    
    Patient Name: John Doe
    Sample: Blood
    
    COMPLETE BLOOD COUNT
    
    Hemoglobin: 12.5 g/dL (Reference: 13.0-17.0) LOW
    RBC Count: 4.5 million/cumm (Reference: 4.5-5.5) NORMAL
    WBC Count: 8500 cells/cumm (Reference: 4000-11000) NORMAL
    Platelet Count: 250000 cells/cumm (Reference: 150000-400000) NORMAL
    
    Blood Sugar Fasting: 126 mg/dL (Reference: 70-100) HIGH
    
    Dr. Smith
    Pathologist
    """
    
    mixed_sample = """
    This document contains various information.
    It has some medical terms like hemoglobin and blood.
    But also mentions agreement and tenant.
    Very ambiguous content here.
    """
    
    print("="*70)
    print("DOMAIN CLASSIFICATION - DEMO")
    print("="*70)
    
    classifier = DomainClassifier()
    
    # Test legal sample
    print("\n📄 LEGAL SAMPLE:")
    print("-"*40)
    result = classifier.classify(legal_sample)
    print(f"   Domain: {result.domain.value.upper()}")
    print(f"   Confidence: {result.confidence:.1%}")
    print(f"   Legal Score: {result.legal_score:.1%}")
    print(f"   Medical Score: {result.medical_score:.1%}")
    print(f"   Top Indicators: {', '.join(result.top_indicators)}")
    
    # Test medical sample
    print("\n🏥 MEDICAL SAMPLE:")
    print("-"*40)
    result = classifier.classify(medical_sample)
    print(f"   Domain: {result.domain.value.upper()}")
    print(f"   Confidence: {result.confidence:.1%}")
    print(f"   Legal Score: {result.legal_score:.1%}")
    print(f"   Medical Score: {result.medical_score:.1%}")
    print(f"   Top Indicators: {', '.join(result.top_indicators)}")
    
    # Test mixed sample
    print("\n❓ MIXED/AMBIGUOUS SAMPLE:")
    print("-"*40)
    result = classifier.classify(mixed_sample)
    print(f"   Domain: {result.domain.value.upper()}")
    print(f"   Confidence: {result.confidence:.1%}")
    print(f"   Legal Score: {result.legal_score:.1%}")
    print(f"   Medical Score: {result.medical_score:.1%}")
    print(f"   Top Indicators: {', '.join(result.top_indicators)}")
    
    print("\n" + "="*70)
    print("DOMAIN TOKENS FOR MODEL:")
    print("="*70)
    print(f"   Legal doc token: {get_domain_token(legal_sample)}")
    print(f"   Medical doc token: {get_domain_token(medical_sample)}")
