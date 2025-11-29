"""
Enhanced Risk Detection Module v2
=================================

Major Improvements:
1. 100+ India-specific risk patterns
2. Context-aware scoring
3. Severity explanation generation
4. Multi-factor risk assessment
5. Cumulative risk scoring

Author: PlainSense Team
Date: November 2025
"""

import numpy as np
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class RiskLevel(Enum):
    """Risk severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class RiskFinding:
    """Individual risk finding"""
    pattern: str
    matched_text: str
    similarity: float
    explanation: str
    suggestion: str


@dataclass 
class RiskAssessment:
    """Complete risk assessment for a clause"""
    level: RiskLevel
    score: float  # 0-1
    findings: List[RiskFinding]
    explanation: str
    suggestions: List[str]
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            'level': self.level.value,
            'score': round(self.score, 3),
            'findings': [
                {
                    'pattern': f.pattern,
                    'explanation': f.explanation,
                    'similarity': round(f.similarity, 2)
                }
                for f in self.findings
            ],
            'explanation': self.explanation,
            'suggestions': self.suggestions,
            'confidence': round(self.confidence, 3)
        }


class EnhancedRiskDetector:
    """
    Enhanced risk detection with India-specific patterns
    """
    
    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 use_gpu: bool = True):
        """
        Initialize risk detector
        
        Args:
            embedding_model: Sentence transformer model
            use_gpu: Use GPU if available
        """
        self.device = "cuda" if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        
        # Load model
        if ML_AVAILABLE:
            print(f"🔄 Loading risk detector: {embedding_model}...")
            self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
            print(f"   ✅ Loaded on {self.device}")
        else:
            self.embedding_model = None
        
        # Initialize risk patterns
        self.risk_patterns = self._init_risk_patterns()
        self.pattern_embeddings = {}
        self._compute_pattern_embeddings()
    
    def _init_risk_patterns(self) -> Dict[RiskLevel, List[Dict]]:
        """
        Initialize comprehensive India-specific risk patterns
        
        Each pattern has:
        - phrase: The risk pattern to match
        - explanation: Plain English explanation
        - suggestion: What the tenant should do
        """
        return {
            # ============================================
            # CRITICAL RISKS - Immediate action required
            # ============================================
            RiskLevel.CRITICAL: [
                {
                    'phrase': 'forfeit entire deposit for any breach',
                    'explanation': 'You could lose your entire security deposit (often 2-3 months rent) for even minor violations.',
                    'suggestion': 'Negotiate specific conditions for deposit forfeiture, not "any breach".'
                },
                {
                    'phrase': 'immediate eviction without notice',
                    'explanation': 'Landlord can ask you to leave immediately without any warning period.',
                    'suggestion': 'Ensure minimum 30 days notice is required for any eviction.'
                },
                {
                    'phrase': 'lease stands cancelled forthwith',
                    'explanation': 'Agreement ends instantly without time to remedy or find alternative accommodation.',
                    'suggestion': 'Request cure period to fix issues before termination.'
                },
                {
                    'phrase': 'vacate premises forthwith',
                    'explanation': 'You must leave immediately without time to pack or find new place.',
                    'suggestion': 'Negotiate minimum 15-30 days to vacate even in breach scenarios.'
                },
                {
                    'phrase': 'surrender vacant possession without objection',
                    'explanation': 'You cannot dispute or challenge when asked to leave.',
                    'suggestion': 'Preserve your right to raise objections through proper channels.'
                },
                {
                    'phrase': 'waive all legal rights and remedies',
                    'explanation': 'You give up your legal protections under Rent Control Act.',
                    'suggestion': 'Remove this clause - it may not be legally enforceable anyway.'
                },
                {
                    'phrase': 'police complaint without notice',
                    'explanation': 'Landlord can file police case against you without warning.',
                    'suggestion': 'Request written notice before any legal action.'
                },
                {
                    'phrase': 'unlimited liability for damages',
                    'explanation': 'You could be liable for unlimited costs beyond your deposit.',
                    'suggestion': 'Cap liability at reasonable amount (e.g., 2x deposit).'
                },
                {
                    'phrase': 'automatic termination on any breach',
                    'explanation': 'Even minor issues like late payment once can end your tenancy.',
                    'suggestion': 'Specify which breaches cause termination and add cure period.'
                },
                {
                    'phrase': 'no refund of deposit under any circumstances',
                    'explanation': 'You will never get your deposit back regardless of condition.',
                    'suggestion': 'This is likely illegal. Deposit must be refunded per law.'
                },
            ],
            
            # ============================================
            # HIGH RISKS - Significant financial impact
            # ============================================
            RiskLevel.HIGH: [
                {
                    'phrase': 'two consecutive months rent default leads to termination',
                    'explanation': 'Missing rent for 2 months causes automatic eviction.',
                    'suggestion': 'Negotiate 3 months with notice before termination.'
                },
                {
                    'phrase': 'penalty of two months rent for early termination',
                    'explanation': 'Leaving before lock-in costs 2 months rent (Rs 30,000-50,000 typically).',
                    'suggestion': 'Negotiate penalty proportional to remaining lock-in period.'
                },
                {
                    'phrase': 'ten percent annual rent escalation',
                    'explanation': '10% yearly increase is above market (standard is 5-7%).',
                    'suggestion': 'Negotiate 5% increase or link to inflation index.'
                },
                {
                    'phrase': 'landlord can enter premises anytime',
                    'explanation': 'No privacy - owner can come without notice.',
                    'suggestion': 'Require 24-48 hours notice except emergencies.'
                },
                {
                    'phrase': 'tenant responsible for all repairs including structural',
                    'explanation': 'You pay for everything including foundation, roof, plumbing issues.',
                    'suggestion': 'Structural repairs are landlord responsibility. Get this corrected.'
                },
                {
                    'phrase': 'interest free deposit for three years',
                    'explanation': 'Large deposit (3-10 lakhs) locked with no interest for 3 years.',
                    'suggestion': 'Negotiate interest at bank FD rate or shorter lock-in.'
                },
                {
                    'phrase': 'deduct from deposit for normal wear and tear',
                    'explanation': 'Routine usage damage charged to you (painting, minor scratches).',
                    'suggestion': 'Normal wear and tear should be landlord cost. Clarify this.'
                },
                {
                    'phrase': 'shall not sublet or have long term guests',
                    'explanation': 'Cannot have family/friends stay for extended periods.',
                    'suggestion': 'Define "long term" clearly (e.g., more than 30 continuous days).'
                },
                {
                    'phrase': 'tenant bears all legal costs in disputes',
                    'explanation': 'You pay landlord\'s lawyer fees even if you win.',
                    'suggestion': 'Each party should bear own costs, or loser pays.'
                },
                {
                    'phrase': 'non refundable maintenance deposit',
                    'explanation': 'Separate deposit that you never get back.',
                    'suggestion': 'All deposits should be refundable. Negotiate removal.'
                },
                {
                    'phrase': 'landlord not liable for any inconvenience',
                    'explanation': 'If water/power issues from landlord side, no compensation.',
                    'suggestion': 'Landlord should provide basic amenities or reduce rent.'
                },
                {
                    'phrase': 'automatic renewal at landlord discretion',
                    'explanation': 'Landlord decides if you can continue, not mutual.',
                    'suggestion': 'Renewal should be mutual consent or tenant option.'
                },
            ],
            
            # ============================================
            # MEDIUM RISKS - Review and understand
            # ============================================
            RiskLevel.MEDIUM: [
                {
                    'phrase': 'three months notice period for termination',
                    'explanation': '3 months notice is long. Standard is 1-2 months.',
                    'suggestion': 'Try negotiating to 2 months notice.'
                },
                {
                    'phrase': 'five percent annual rent increase',
                    'explanation': '5% annual increase is market standard in India.',
                    'suggestion': 'This is reasonable but ensure it\'s compounded, not on original rent.'
                },
                {
                    'phrase': 'painting and deep cleaning at tenant cost when vacating',
                    'explanation': 'Rs 15,000-30,000 for painting at end of tenancy.',
                    'suggestion': 'Reasonable if tenancy > 2 years. Document initial condition.'
                },
                {
                    'phrase': 'maintenance charges extra at actuals',
                    'explanation': 'Society maintenance on top of rent (Rs 2,000-5,000/month).',
                    'suggestion': 'Get estimate of monthly maintenance. Factor in total cost.'
                },
                {
                    'phrase': 'separate meter reading for electricity',
                    'explanation': 'You pay EB at actual consumption.',
                    'suggestion': 'Standard practice. Verify meter is working and take initial reading.'
                },
                {
                    'phrase': 'no pets allowed on premises',
                    'explanation': 'Cannot keep dogs, cats or other pets.',
                    'suggestion': 'If you have pets, negotiate before signing or find another place.'
                },
                {
                    'phrase': 'restrict number of occupants',
                    'explanation': 'Limit on how many people can live in the flat.',
                    'suggestion': 'Ensure limit is reasonable for flat size (usually 2-3 per bedroom).'
                },
                {
                    'phrase': 'landlord can show flat to prospective tenants',
                    'explanation': 'During notice period, strangers may visit your home.',
                    'suggestion': 'Limit viewings to specific hours with advance notice.'
                },
                {
                    'phrase': 'lock in period of eleven months',
                    'explanation': 'Cannot leave within 11 months without penalty.',
                    'suggestion': '6 months lock-in is more reasonable. Negotiate down.'
                },
                {
                    'phrase': 'must obtain written permission for modifications',
                    'explanation': 'Cannot even put nails in wall without landlord okay.',
                    'suggestion': 'Get blanket permission for minor modifications like curtain rods.'
                },
            ],
            
            # ============================================
            # LOW RISKS - Standard clauses
            # ============================================
            RiskLevel.LOW: [
                {
                    'phrase': 'residential purpose only',
                    'explanation': 'Cannot run business from this premises.',
                    'suggestion': 'Standard clause. Fine for pure residential use.'
                },
                {
                    'phrase': 'rent payable by fifth of every month',
                    'explanation': 'Monthly rent due date.',
                    'suggestion': 'Standard practice. Set reminder for timely payment.'
                },
                {
                    'phrase': 'one month notice for termination',
                    'explanation': '30 days notice is reasonable and standard.',
                    'suggestion': 'This is fair for both parties.'
                },
                {
                    'phrase': 'maintain premises in good condition',
                    'explanation': 'Keep the property clean and don\'t damage it.',
                    'suggestion': 'Reasonable expectation. Document initial condition with photos.'
                },
                {
                    'phrase': 'return keys and possession on termination',
                    'explanation': 'Hand over keys when leaving.',
                    'suggestion': 'Standard. Get receipt for key return.'
                },
                {
                    'phrase': 'agreement registered with sub registrar',
                    'explanation': 'Legally registered agreement.',
                    'suggestion': 'Good for you - provides legal protection.'
                },
                {
                    'phrase': 'governed by laws of Karnataka',
                    'explanation': 'Local state law applies to disputes.',
                    'suggestion': 'Standard. Know your state\'s Rent Control Act.'
                },
                {
                    'phrase': 'quiet and peaceful enjoyment of premises',
                    'explanation': 'Your right to live peacefully.',
                    'suggestion': 'Good clause - protects your privacy.'
                },
            ]
        }
    
    def _compute_pattern_embeddings(self):
        """Pre-compute embeddings for all patterns"""
        if not self.embedding_model:
            return
        
        print("   Computing risk pattern embeddings...")
        
        for level, patterns in self.risk_patterns.items():
            phrases = [p['phrase'] for p in patterns]
            embeddings = self.embedding_model.encode(phrases)
            self.pattern_embeddings[level] = embeddings
    
    def _keyword_risk_boost(self, text: str) -> Dict[str, float]:
        """
        Keyword-based risk boosting
        
        Returns multipliers for each risk level
        """
        text_lower = text.lower()
        
        boosts = {
            RiskLevel.CRITICAL: 1.0,
            RiskLevel.HIGH: 1.0,
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.LOW: 1.0
        }
        
        # Critical keywords
        critical_keywords = [
            'forfeit', 'forthwith', 'immediately', 'waive', 'unlimited',
            'unconditional', 'absolute', 'irrevocable', 'no recourse'
        ]
        
        # High risk keywords
        high_keywords = [
            'penalty', 'liable', 'damages', 'compensation', 'breach',
            'default', 'terminate', 'evict', 'legal action', 'court'
        ]
        
        # Check for keywords
        for kw in critical_keywords:
            if kw in text_lower:
                boosts[RiskLevel.CRITICAL] *= 1.15
                
        for kw in high_keywords:
            if kw in text_lower:
                boosts[RiskLevel.HIGH] *= 1.1
        
        return boosts
    
    def _keyword_risk_detection(self, text: str) -> Tuple[List[RiskFinding], float]:
        """
        Pure keyword-based risk detection fallback
        
        Returns:
            (findings, max_score)
        """
        text_lower = text.lower()
        findings = []
        max_score = 0.0
        
        # Keywords with associated risk levels and scores
        # More flexible patterns
        keyword_patterns = {
            RiskLevel.CRITICAL: {
                'forfeit entire': 0.95,
                'forfeit security deposit': 0.93,
                'forfeit deposit': 0.90,
                'immediate eviction': 0.92,
                'eviction without notice': 0.90,
                'vacate forthwith': 0.88,
                'vacate immediately': 0.88,
                'shall stand cancelled forthwith': 0.90,
                'forthwith': 0.85,
                'waive all legal rights': 0.95,
                'waive legal rights': 0.90,
                'police complaint': 0.85,
                'no refund': 0.88,
                'unlimited liability': 0.90,
                'automatic termination': 0.85,
                'irrevocable': 0.82,
                'unconditional forfeiture': 0.92,
                'any breach': 0.85,  # "for any breach"
            },
            RiskLevel.HIGH: {
                'penalty of two months': 0.80,
                'two months penalty': 0.80,
                'penalty for early termination': 0.75,
                'early termination penalty': 0.75,
                '10% annual': 0.78,
                '10 percent': 0.76,
                'annual rent escalation of 10': 0.80,
                'escalation of 10%': 0.78,
                '10% escalation': 0.78,
                'ten percent': 0.72,
                'liable for damages': 0.75,
                'entire period of lease': 0.70,
                'lock in period': 0.68,
                'lock-in period': 0.68,
                'consecutive months': 0.70,  # rent default
                'two consecutive': 0.72,
                'eviction': 0.70,
            },
            RiskLevel.MEDIUM: {
                'maintenance charges': 0.55,
                'repairs at tenant cost': 0.60,
                'tenant shall bear': 0.55,
                'society charges': 0.50,
                'permission from landlord': 0.52,
                'prior written consent': 0.50,
                'subletting prohibited': 0.55,
                'no pets allowed': 0.48,
                'inspection': 0.45,
                'periodic visits': 0.45,
                'three months notice': 0.55,
                '3 months notice': 0.55,
                'notice period required': 0.52,
                'breach': 0.50,  # moved to medium when standalone
                'default': 0.48,
                'terminate': 0.45,
            },
            RiskLevel.LOW: {
                'monthly rent': 0.38,
                'rent payable': 0.40,
                'advance rent': 0.42,
                'due on': 0.38,
                'payable on': 0.35,
                'refundable deposit': 0.35,
                'notice period': 0.40,
                'normal wear and tear': 0.30,
                'on or before': 0.32,
            }
        }
        
        # Check each pattern
        for level, patterns in keyword_patterns.items():
            for phrase, score in patterns.items():
                if phrase in text_lower:
                    # Get the matching pattern details from risk_patterns
                    pattern_info = None
                    for p in self.risk_patterns.get(level, []):
                        if any(kw in p['phrase'].lower() for kw in phrase.split()):
                            pattern_info = p
                            break
                    
                    if pattern_info:
                        findings.append(RiskFinding(
                            pattern=phrase,
                            matched_text=text[:200],
                            similarity=score,
                            explanation=pattern_info['explanation'],
                            suggestion=pattern_info['suggestion']
                        ))
                    else:
                        # Create specific explanations based on keyword
                        explanations = {
                            'forfeit': ('Risk of losing deposit money', 'Negotiate specific conditions for forfeiture'),
                            'forthwith': ('Immediate action required without notice', 'Request minimum notice period'),
                            'eviction': ('Risk of forced removal from property', 'Ensure proper notice requirements'),
                            'breach': ('Violation of agreement terms', 'Clarify what constitutes breach'),
                            'penalty': ('Financial penalty clause', 'Negotiate penalty amounts'),
                            'escalation': ('Rent increase clause', 'Cap escalation at 5%'),
                            '10%': ('High rent escalation rate', 'Standard is 5-7% annual'),
                            'notice': ('Notice period requirement', 'Standard notice period'),
                            'terminate': ('Agreement termination clause', 'Review termination conditions'),
                            'consecutive': ('Consecutive payment default', 'Negotiate cure period'),
                        }
                        
                        exp, sug = 'Potentially risky clause', 'Review carefully'
                        for key, (e, s) in explanations.items():
                            if key in phrase:
                                exp, sug = e, s
                                break
                        
                        findings.append(RiskFinding(
                            pattern=phrase,
                            matched_text=text[:200],
                            similarity=score,
                            explanation=exp,
                            suggestion=sug
                        ))
                    
                    if score > max_score:
                        max_score = score
        
        return findings, max_score
    
    def assess_clause(self, 
                      text: str,
                      threshold: float = 0.45) -> RiskAssessment:
        """
        Assess risk of a single clause
        
        Args:
            text: Clause text
            threshold: Similarity threshold for matching
            
        Returns:
            RiskAssessment
        """
        findings = []
        
        # Fallback to keyword-only mode if ML not available
        if not self.embedding_model:
            findings, max_score = self._keyword_risk_detection(text)
            
            if not findings:
                return RiskAssessment(
                    level=RiskLevel.NONE,
                    score=0.0,
                    findings=[],
                    explanation="No significant risks identified.",
                    suggestions=["This clause appears standard."],
                    confidence=0.6
                )
            
            # Determine level from findings
            level_priority = {
                RiskLevel.CRITICAL: 4,
                RiskLevel.HIGH: 3,
                RiskLevel.MEDIUM: 2,
                RiskLevel.LOW: 1,
                RiskLevel.NONE: 0
            }
            
            max_level = RiskLevel.LOW
            for finding in findings:
                for lvl, patterns in {
                    RiskLevel.CRITICAL: ['forfeit', 'forthwith', 'waive', 'immediate', 'no refund', 'unlimited', 'irrevocable'],
                    RiskLevel.HIGH: ['penalty', 'termination', 'breach', 'default', 'liable', 'eviction', 'lock'],
                    RiskLevel.MEDIUM: ['maintenance', 'tenant shall', 'permission', 'consent', 'prohibited'],
                    RiskLevel.LOW: ['rent', 'deposit', 'notice']
                }.items():
                    if any(p in finding.pattern for p in patterns):
                        if level_priority.get(lvl, 0) > level_priority.get(max_level, 0):
                            max_level = lvl
                        break
            
            return RiskAssessment(
                level=max_level,
                score=max_score,
                findings=findings,
                explanation=" | ".join([f.explanation for f in findings[:3]]),
                suggestions=[f.suggestion for f in findings[:3]],
                confidence=0.65
            )
        
        # Get clause embedding
        clause_embedding = self.embedding_model.encode(text)
        
        # Get keyword boosts
        boosts = self._keyword_risk_boost(text)
        
        # Check against all patterns
        for level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]:
            if level not in self.pattern_embeddings:
                continue
            
            # Compute similarities
            similarities = cosine_similarity(
                [clause_embedding],
                self.pattern_embeddings[level]
            )[0]
            
            # Apply boost
            similarities = similarities * boosts[level]
            
            # Find matches above threshold
            patterns = self.risk_patterns[level]
            for idx, sim in enumerate(similarities):
                if sim > threshold:
                    pattern = patterns[idx]
                    findings.append(RiskFinding(
                        pattern=pattern['phrase'],
                        matched_text=text[:200],
                        similarity=float(sim),
                        explanation=pattern['explanation'],
                        suggestion=pattern['suggestion']
                    ))
        
        # Determine overall risk level
        if not findings:
            return RiskAssessment(
                level=RiskLevel.NONE,
                score=0.0,
                findings=[],
                explanation="No significant risks identified in this clause.",
                suggestions=["This clause appears standard."],
                confidence=0.8
            )
        
        # Sort findings by severity and similarity
        level_priority = {
            RiskLevel.CRITICAL: 4,
            RiskLevel.HIGH: 3,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 1
        }
        
        # Get highest risk level found
        max_level = RiskLevel.NONE
        max_score = 0.0
        
        for finding in findings:
            # Determine level of this finding
            for level, patterns in self.risk_patterns.items():
                if any(p['phrase'] == finding.pattern for p in patterns):
                    if level_priority.get(level, 0) > level_priority.get(max_level, 0):
                        max_level = level
                    if finding.similarity > max_score:
                        max_score = finding.similarity
                    break
        
        # Generate overall explanation
        explanations = [f.explanation for f in findings[:3]]
        suggestions = [f.suggestion for f in findings[:3]]
        
        return RiskAssessment(
            level=max_level,
            score=min(max_score, 1.0),
            findings=findings,
            explanation=" | ".join(explanations),
            suggestions=suggestions,
            confidence=min(max_score + 0.1, 1.0)
        )
    
    def assess_document(self, 
                        clauses: List[Dict],
                        threshold: float = 0.45) -> Dict:
        """
        Assess risk across entire document
        
        Args:
            clauses: List of clause dictionaries with 'content' key
            threshold: Similarity threshold
            
        Returns:
            Document-level risk assessment
        """
        print(f"\n🔍 Assessing document risk ({len(clauses)} clauses)")
        
        clause_assessments = []
        risk_distribution = {level.value: 0 for level in RiskLevel}
        all_findings = []
        
        for clause in clauses:
            content = clause.get('content', clause.get('text', ''))
            if not content:
                continue
            
            assessment = self.assess_clause(content, threshold)
            clause_assessments.append({
                'clause_id': clause.get('id', 0),
                'assessment': assessment.to_dict()
            })
            
            risk_distribution[assessment.level.value] += 1
            all_findings.extend(assessment.findings)
        
        # Determine document risk level
        if risk_distribution['critical'] > 0:
            doc_level = RiskLevel.CRITICAL
        elif risk_distribution['high'] >= 2:
            doc_level = RiskLevel.CRITICAL
        elif risk_distribution['high'] > 0:
            doc_level = RiskLevel.HIGH
        elif risk_distribution['medium'] >= 3:
            doc_level = RiskLevel.HIGH
        elif risk_distribution['medium'] > 0:
            doc_level = RiskLevel.MEDIUM
        else:
            doc_level = RiskLevel.LOW
        
        # Calculate average risk score
        scores = [a['assessment']['score'] for a in clause_assessments]
        avg_score = np.mean(scores) if scores else 0.0
        
        # Count high risk clauses
        high_risk_count = (
            risk_distribution['critical'] + 
            risk_distribution['high']
        )
        
        print(f"   Document Risk Level: {doc_level.value.upper()}")
        print(f"   Average Risk Score: {avg_score:.3f}")
        print(f"   High-risk clauses: {high_risk_count}")
        print(f"\n   Risk Distribution:")
        for level, count in risk_distribution.items():
            if count > 0:
                print(f"      {level}: {count}")
        
        # Get top findings
        top_findings = sorted(
            all_findings, 
            key=lambda f: f.similarity, 
            reverse=True
        )[:5]
        
        if top_findings:
            print(f"\n   ⚠️ Top Risk Findings:")
            for f in top_findings:
                print(f"      - {f.pattern[:60]}...")
                print(f"        Risk: {f.similarity:.2f}")
        
        return {
            'document_risk_level': doc_level.value,
            'average_risk_score': float(avg_score),
            'high_risk_clause_count': high_risk_count,
            'total_clauses': len(clauses),
            'risk_distribution': risk_distribution,
            'clause_assessments': clause_assessments,
            'top_findings': [
                {
                    'pattern': f.pattern,
                    'explanation': f.explanation,
                    'suggestion': f.suggestion,
                    'similarity': f.similarity
                }
                for f in top_findings
            ]
        }


# Backward compatibility
MLRiskDetector = EnhancedRiskDetector


if __name__ == "__main__":
    # Test
    detector = EnhancedRiskDetector()
    
    test_clauses = [
        {
            'id': 1,
            'content': 'If tenant fails to pay rent for two consecutive months, the lease shall stand cancelled forthwith and tenant must vacate immediately.'
        },
        {
            'id': 2,
            'content': 'Monthly rent of Rs. 25,000 payable on or before 5th of every month.'
        },
        {
            'id': 3,
            'content': 'Tenant shall forfeit entire security deposit for any breach of this agreement.'
        },
        {
            'id': 4,
            'content': 'Annual rent escalation of 10% applicable from second year.'
        },
    ]
    
    result = detector.assess_document(test_clauses)
    print(f"\n\nFinal Assessment: {result['document_risk_level'].upper()}")
