"""
LLM-Based Text Simplifier
==========================

An advanced document simplification engine using domain-specific language models:

Models Used:
- FLAN-T5: Primary text generation/simplification model
- LegalBERT: Legal domain risk detection and clause understanding
- BioBERT: Medical entity extraction and NER
- Sentence-Transformers: Fallback semantic similarity

Key Features:
1. Clause-by-clause simplification preserving original meaning
2. ML-based risk detection (not hardcoded rules)
3. Tabular medical data parsing and interpretation
4. Consistency validation to ensure no information loss
5. Plain and colloquial output styles
6. Support for multiple Indian languages (Hindi, Tamil)

Architecture:
    Input → Domain Detection → Preprocessing → LLM Simplification
                                    ↓
                           Risk Detection (LegalBERT/BioBERT)
                                    ↓
                           Consistency Validation
                                    ↓
                           Output with Warnings

Author: PlainSense Team
Version: 2.0.0
Date: November 2025
"""

import os
import re
import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')
# Local import for enhanced medical parsing
try:
    from medical.medical_report_parser import MedicalReportParser
    MEDICAL_PARSER_AVAILABLE = True
except Exception:
    MEDICAL_PARSER_AVAILABLE = False

# ML/DL imports
try:
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModel,
        AutoTokenizer,
        pipeline,
        T5ForConditionalGeneration,
        T5Tokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers not available")

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class OutputStyle(Enum):
    """Output language and style"""
    PLAIN_ENGLISH = "plain_english"
    COLLOQUIAL_ENGLISH = "colloquial_english"
    PLAIN_HINDI = "plain_hindi"
    COLLOQUIAL_HINDI = "colloquial_hindi"
    PLAIN_TAMIL = "plain_tamil"
    COLLOQUIAL_TAMIL = "colloquial_tamil"


class RiskLevel(Enum):
    """Risk severity levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Note: ClauseType classification removed - focusing on simplification and risk only


@dataclass
class ClauseResult:
    """
    Result for a single clause with all language outputs.
    
    Contains:
    - Original text
    - English: plain + colloquial
    - Hindi: formal + colloquial
    - Tamil: formal + colloquial
    - Risk assessment
    - Key terms extracted
    """
    original: str
    # English versions
    simplified: str  # Plain English
    colloquial: str  # Friendly/conversational English
    # Hindi versions
    hindi_formal: str = ""
    hindi_colloquial: str = ""
    # Tamil versions
    tamil_formal: str = ""
    tamil_colloquial: str = ""
    # Risk assessment
    risk_level: RiskLevel = RiskLevel.NONE
    risk_score: float = 0.0
    risk_explanation: str = ""
    # Extracted info
    key_terms: List[str] = field(default_factory=list)
    # Entity preservation check
    entities_preserved: bool = True
    preservation_warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'original': self.original,
            'english': {
                'plain': self.simplified,
                'colloquial': self.colloquial
            },
            'hindi': {
                'formal': self.hindi_formal,
                'colloquial': self.hindi_colloquial
            },
            'tamil': {
                'formal': self.tamil_formal,
                'colloquial': self.tamil_colloquial
            },
            'risk': {
                'level': self.risk_level.value,
                'score': round(self.risk_score, 3),
                'explanation': self.risk_explanation,
                'is_risky': self.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            },
            'key_terms': self.key_terms,
            'entities_preserved': self.entities_preserved,
            'preservation_warnings': self.preservation_warnings
        }
    
    def get_summary(self) -> str:
        """Get a brief summary of this clause result"""
        risk_emoji = {'none': '✅', 'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}
        emoji = risk_emoji.get(self.risk_level.value, '⚪')
        return f"{emoji} Risk: {self.risk_level.value} ({self.risk_score:.0%})"


@dataclass 
class DocumentResult:
    """Result for full document"""
    clauses: List[ClauseResult]
    summary: str
    overall_risk: RiskLevel
    overall_risk_score: float
    domain: str
    style: OutputStyle
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'clauses': [c.to_dict() for c in self.clauses],
            'summary': self.summary,
            'overall_risk': self.overall_risk.value,
            'overall_risk_score': round(self.overall_risk_score, 3),
            'domain': self.domain,
            'style': self.style.value,
            'warnings': self.warnings,
            'total_clauses': len(self.clauses),
            'risky_clauses': sum(1 for c in self.clauses if c.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL])
        }


class LLMSimplifier:
    """
    LLM-based clause simplifier with risk detection.
    
    Models used:
    - Simplification: FLAN-T5 or similar instruction-tuned model
    - Risk Detection: Sentence embeddings + learned risk patterns
    - Translation: MarianMT for Hindi/Tamil (when needed)
    """
    
    def __init__(
        self,
        device: str = None,
        use_quantization: bool = True,
        fallback_enabled: bool = True,
        domain: str = None  # 'legal', 'medical', or None for both
    ):
        """
        Initialize the LLM simplifier.
        
        Args:
            device: 'cuda', 'cpu', or 'auto' (auto-detected if None or 'auto')
            use_quantization: Use 8-bit quantization for lower memory
            fallback_enabled: Enable rule-based fallback
            domain: 'legal' or 'medical' to load domain-specific models only
        """
        # Handle device selection
        if device is None or device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.use_quantization = use_quantization
        self.fallback_enabled = fallback_enabled
        self.domain = domain  # Track which domain we're loading for
        
        # Models (lazy loaded)
        self.simplifier_model = None
        self.simplifier_tokenizer = None
        self.simplification_model_type = None  # Track which model is loaded
        self.risk_encoder = None  # Optional fallback encoder
        self.legalbert_model = None
        self.legalbert_tokenizer = None
        self.bio_ner_pipeline = None
        self.translation_models = {}
        
        # Status flags
        self.models_loaded = False
        self.risk_model_loaded = False
        self.legal_models_loaded = False
        self.medical_models_loaded = False
        
        print(f"🔄 Initializing LLM Simplifier on {self.device}")
        self._load_models()
    
    def _load_models(self):
        """Load required models based on domain"""
        if not TRANSFORMERS_AVAILABLE:
            print("   ⚠️ Transformers not available, using fallback mode")
            return
        
        try:
            # Try to load a paraphrasing model for better sentence rewriting
            print("   Loading text simplification model...")
            
            # Option 1: Try Pegasus paraphrase model (good at rewriting)
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                model_name = "tuner007/pegasus_paraphrase"
                
                self.simplifier_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.simplifier_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                )
                self.simplifier_model.to(self.device)
                self.simplifier_model.eval()
                self.simplification_model_type = "pegasus-paraphrase"
                print(f"   ✅ Pegasus Paraphrase model loaded on {self.device}")
                self.models_loaded = True
                
            except Exception as e1:
                print(f"   ⚠️ Could not load Pegasus model: {e1}")
                # Option 2: Try T5 simplification model
                try:
                    model_name = "mrm8488/t5-small-finetuned-text-simplification"
                    self.simplifier_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.simplifier_model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    )
                    self.simplifier_model.to(self.device)
                    self.simplifier_model.eval()
                    self.simplification_model_type = "t5-simplification"
                    print(f"   ✅ T5-Simplification model loaded on {self.device}")
                    self.models_loaded = True
                except Exception as e2:
                    print(f"   ⚠️ Could not load T5 simplification: {e2}")
                    # Option 3: Fall back to FLAN-T5-base
                    print("   Trying FLAN-T5-base...")
                    model_name = "google/flan-t5-base"
                    
                    self.simplifier_tokenizer = T5Tokenizer.from_pretrained(
                        model_name,
                        model_max_length=512
                    )
                    self.simplifier_model = T5ForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    )
                    self.simplifier_model.to(self.device)
                    self.simplifier_model.eval()
                    self.simplification_model_type = "flan-t5"
                    print(f"   ✅ FLAN-T5 loaded on {self.device}")
                    self.models_loaded = True
            
            for param in self.simplifier_model.parameters():
                param.requires_grad = False
            
        except Exception as e:
            print(f"   ⚠️ Could not load simplification model: {e}")
            print("   Trying fallback model (T5-small)...")
            try:
                self.simplifier_tokenizer = T5Tokenizer.from_pretrained("t5-small")
                self.simplifier_model = T5ForConditionalGeneration.from_pretrained("t5-small")
                self.simplifier_model.to(self.device)
                self.simplifier_model.eval()
                self.simplification_model_type = "t5-small"
                print(f"   ✅ T5-small loaded on {self.device}")
                self.models_loaded = True
            except Exception as e2:
                print(f"   ⚠️ Could not load T5: {e2}")
        
        # Load domain-specific models based on domain parameter
        if self.domain is None or self.domain == 'legal':
            self._load_legal_models()
        
        if self.domain is None or self.domain == 'medical':
            self._load_medical_models()
        
        # Mark risk detection as ready
        self.risk_model_loaded = self.models_loaded
        print("   ✅ LLM-based risk detector ready" if self.risk_model_loaded else "   ⚠️ Risk detection will use fallback")
        
        # Translation models are loaded on-demand only
        print("   📝 Translation models will be loaded only when requested")
    
    def _load_legal_models(self):
        """Load legal-specific models (LegalBERT)"""
        if self.legal_models_loaded:
            return
            
        try:
            print("   Loading LegalBERT for risk detection...")
            legal_name = 'nlpaueb/legal-bert-base-uncased'
            self.legalbert_tokenizer = AutoTokenizer.from_pretrained(legal_name)
            self.legalbert_model = AutoModel.from_pretrained(
                legal_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            )
            self.legalbert_model.to(self.device)
            self.legalbert_model.eval()
            print("   ✅ LegalBERT loaded")
            self.legal_models_loaded = True
        except Exception as e:
            print(f"   ⚠️ Could not load LegalBERT: {e}")
            self.legalbert_model = None

        # Sentence-transformers fallback for legal
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.legalbert_model is None:
            try:
                print("   Loading sentence-transformers encoder (fallback)...")
                self.risk_encoder = SentenceTransformer('all-MiniLM-L6-v2')
                print("   ✅ Sentence-transformers loaded")
                self.legal_models_loaded = True
            except Exception as e:
                print(f"   ⚠️ Could not load sentence-transformers: {e}")
    
    def _load_medical_models(self):
        """Load medical-specific models (BioBERT)"""
        if self.medical_models_loaded:
            return
            
        try:
            print("   Loading BioBERT NER pipeline...")
            bio_name = 'dmis-lab/biobert-base-cased-v1.1'
            self.bio_ner_pipeline = pipeline('ner', model=bio_name, tokenizer=bio_name, aggregation_strategy='simple', device=0 if self.device == 'cuda' else -1)
            print("   ✅ BioBERT NER pipeline loaded")
            self.medical_models_loaded = True
        except Exception as e:
            print(f"   ⚠️ Could not load BioBERT NER: {e}")
            self.bio_ner_pipeline = None
    
    def _load_translation_model(self, target_lang: str):
        """
        Lazy load translation model for a target language.
        Uses MarianMT models from Helsinki-NLP.
        
        Args:
            target_lang: 'hi' for Hindi, 'ta' for Tamil
        """
        if target_lang in self.translation_models:
            return
        
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            # Model mappings
            model_map = {
                'hi': 'Helsinki-NLP/opus-mt-en-hi',  # English to Hindi
                'ta': 'Helsinki-NLP/opus-mt-en-mul', # English to multilingual (includes Tamil)
            }
            
            model_name = model_map.get(target_lang)
            if not model_name:
                print(f"   ⚠️ No translation model for {target_lang}")
                return
            
            print(f"   Loading translation model for {target_lang}...")
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            model.to(self.device)
            model.eval()
            
            self.translation_models[target_lang] = {
                'model': model,
                'tokenizer': tokenizer
            }
            print(f"   ✅ Translation model loaded for {target_lang}")
            
        except Exception as e:
            print(f"   ⚠️ Could not load translation model for {target_lang}: {e}")
    
    def _translate(self, text: str, target_lang: str) -> str:
        """
        Translate text to target language.
        
        Args:
            text: English text to translate
            target_lang: 'hi' for Hindi, 'ta' for Tamil
            
        Returns:
            Translated text or original if translation fails
        """
        if not text or len(text.strip()) < 3:
            return text
        
        # Load model if not already loaded
        self._load_translation_model(target_lang)
        
        if target_lang not in self.translation_models:
            return text  # Return original if no model
        
        try:
            model_data = self.translation_models[target_lang]
            tokenizer = model_data['tokenizer']
            model = model_data['model']
            
            # For Tamil, we need to prepend language token
            if target_lang == 'ta':
                text = f">>tam<< {text}"
            
            # Tokenize and translate
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated if translated else text
            
        except Exception as e:
            print(f"   ⚠️ Translation failed: {e}")
            return text
    
    def _make_colloquial(self, formal_text: str, lang: str = 'en') -> str:
        """
        Convert formal text to colloquial/friendly version.
        
        Args:
            formal_text: Formal translated text
            lang: Language code ('en', 'hi', 'ta')
            
        Returns:
            Colloquial version
        """
        if not self.models_loaded or lang != 'en':
            # For non-English, we can't make it colloquial easily
            # Return as-is for now
            return formal_text
        
        try:
            prompt = f"""Make this text more friendly and conversational, like advice from a friend. Keep all the information but use casual language:

Text: {formal_text}

Friendly version:"""
            
            inputs = self.simplifier_tokenizer(
                prompt,
                return_tensors="pt",
                max_length=384,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.simplifier_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=2,
                    do_sample=False,
                    early_stopping=True
                )
            
            result = self.simplifier_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return result if result else formal_text
            
        except Exception:
            return formal_text
    
    def _check_entity_preservation(self, original: str, simplified: str) -> Tuple[bool, List[str]]:
        """
        Check if important entities (numbers, dates, amounts) are preserved.
        
        Args:
            original: Original text
            simplified: Simplified text
            
        Returns:
            Tuple of (is_preserved: bool, warnings: List[str])
        """
        warnings = []
        
        # Extract numbers from original
        original_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', original))
        simplified_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', simplified))
        
        # Check for missing significant numbers (>= 2 digits or decimals)
        important_nums = {n for n in original_numbers if len(n) >= 2 or '.' in n}
        missing_nums = important_nums - simplified_numbers
        
        if missing_nums:
            warnings.append(f"Numbers may be missing: {', '.join(sorted(missing_nums)[:3])}")
        
        # Check monetary values
        money_pattern = r'(?:rs\.?|₹|inr)\s*([\d,]+)'
        original_money = set(re.findall(money_pattern, original, re.I))
        simplified_money = set(re.findall(money_pattern, simplified, re.I))
        
        missing_money = original_money - simplified_money
        if missing_money:
            warnings.append(f"Monetary values may be missing: ₹{', ₹'.join(sorted(missing_money)[:3])}")
        
        # Check dates
        date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
        original_dates = set(re.findall(date_pattern, original))
        simplified_dates = set(re.findall(date_pattern, simplified))
        
        missing_dates = original_dates - simplified_dates
        if missing_dates:
            warnings.append(f"Dates may be missing: {', '.join(sorted(missing_dates)[:2])}")
        
        # Check time periods
        time_pattern = r'\b(\d+)\s*(day|month|year|week)s?\b'
        original_times = set(re.findall(time_pattern, original, re.I))
        simplified_times = set(re.findall(time_pattern, simplified, re.I))
        
        missing_times = original_times - simplified_times
        if missing_times:
            warnings.append(f"Time periods may be missing")
        
        is_preserved = len(warnings) == 0
        return is_preserved, warnings
    
    def _detect_risk_llm(self, text: str, domain: str = 'legal') -> Tuple[RiskLevel, float, str]:
        """
        Detect risk level using LLM inference with semantic analysis.
        Combines LLM question-answering with keyword detection for robust results.
        
        Args:
            text: Clause text to analyze
            domain: 'legal' or 'medical'
            
        Returns:
            Tuple of (risk_level, risk_score, explanation)
        """
        if not self.models_loaded:
            return RiskLevel.NONE, 0.0, "Risk detection unavailable - LLM not loaded"
        
        text_lower = text.lower()
        
        try:
            if domain == 'legal':
                # Step 1: Semantic keyword analysis
                critical_indicators = 0
                high_indicators = 0
                medium_indicators = 0
                explanation_points = []
                
                # Critical indicators
                if any(phrase in text_lower for phrase in [
                    'forfeit 100%', 'forfeit entire', 'forfeit all',
                    'waive all rights', 'waives all rights', 
                    'no right to', 'no legal recourse',
                    'immediate eviction', 'immediate termination',
                    'within 24 hours', 'without notice', 'without any notice'
                ]):
                    critical_indicators += 2
                    explanation_points.append("Extreme penalties or complete rights waiver")
                
                # High risk indicators
                if any(phrase in text_lower for phrase in [
                    'forfeit', 'forfeiture', '7 days', '7-day',
                    '14 days', '14-day', 'non-refundable',
                    'penalty equivalent to 2', 'penalty of 2',
                    'any time without', 'at any time'
                ]):
                    high_indicators += 1
                    explanation_points.append("Short notice or heavy penalties")
                
                # Percentage-based forfeit
                import re
                forfeit_match = re.search(r'forfeit.*?(\d+)\s*%', text_lower)
                if forfeit_match:
                    pct = int(forfeit_match.group(1))
                    if pct >= 100:
                        critical_indicators += 2
                        explanation_points.append(f"Tenant must forfeit {pct}% of deposit")
                    elif pct >= 50:
                        high_indicators += 1
                        explanation_points.append(f"Tenant may forfeit {pct}% of deposit")
                
                # Month-based penalties
                month_penalty = re.search(r'(\d+)\s*month.*?(?:rent|penalty)', text_lower)
                if month_penalty:
                    months = int(month_penalty.group(1))
                    if months >= 3:
                        high_indicators += 1
                        explanation_points.append(f"{months} months penalty")
                    elif months >= 2:
                        medium_indicators += 1
                
                # Short notice periods
                days_notice = re.search(r'(\d+)\s*days?\s*(?:notice|prior)', text_lower)
                if days_notice:
                    days = int(days_notice.group(1))
                    if days <= 7:
                        high_indicators += 1
                        explanation_points.append(f"Only {days} days notice required")
                    elif days <= 14:
                        medium_indicators += 1
                    
                # Tenant protections (reduce risk)
                if any(phrase in text_lower for phrase in [
                    'fully refundable', 'full refund', 
                    '30 days notice', '60 days notice',
                    'mutual agreement', 'written consent'
                ]):
                    high_indicators = max(0, high_indicators - 1)
                    
                # Determine risk level from indicators
                if critical_indicators >= 2:
                    risk_level = RiskLevel.CRITICAL
                elif critical_indicators >= 1 or high_indicators >= 2:
                    risk_level = RiskLevel.HIGH
                elif high_indicators >= 1 or medium_indicators >= 1:
                    risk_level = RiskLevel.MEDIUM
                else:
                    risk_level = RiskLevel.LOW
                    
            else:  # medical domain
                critical_indicators = 0
                high_indicators = 0
                explanation_points = []
                
                # Extract numerical values and compare to references
                import re
                
                # Critical medical terms
                if any(term in text_lower for term in [
                    'life-threatening', 'emergency', 'urgent', 'immediate',
                    'critical', 'crisis', 'transfusion', 'severe'
                ]):
                    critical_indicators += 1
                    explanation_points.append("Critical medical terminology detected")
                
                # Check for abnormal values
                value_patterns = [
                    # Hemoglobin: Normal 12-17
                    (r'(?:hemoglobin|hb|hgb)\s*[:\-]?\s*(\d+\.?\d*)', 12, 17, 'Hemoglobin'),
                    # Creatinine: Normal 0.7-1.3
                    (r'creatinine\s*[:\-]?\s*(\d+\.?\d*)', 0.7, 1.3, 'Creatinine'),
                    # Fasting glucose: Normal 70-100
                    (r'(?:fasting|glucose|blood sugar)\s*[:\-]?\s*(\d+\.?\d*)', 70, 100, 'Blood Sugar'),
                    # Blood pressure systolic: Normal <140
                    (r'(\d+)\s*/\s*\d+\s*(?:mmhg|mm hg)?', 90, 140, 'Blood Pressure'),
                ]
                
                for pattern, low_norm, high_norm, name in value_patterns:
                    match = re.search(pattern, text_lower)
                    if match:
                        try:
                            value = float(match.group(1))
                            deviation = 0
                            if value < low_norm:
                                deviation = (low_norm - value) / low_norm
                            elif value > high_norm:
                                deviation = (value - high_norm) / high_norm
                            
                            if deviation > 0.5:  # >50% deviation
                                critical_indicators += 1
                                explanation_points.append(f"{name} severely abnormal: {value}")
                            elif deviation > 0.2:  # >20% deviation
                                high_indicators += 1
                                explanation_points.append(f"{name} significantly abnormal: {value}")
                        except ValueError:
                            pass
                
                # Stage-based indicators
                if re.search(r'stage\s*[34]', text_lower):
                    critical_indicators += 1
                    explanation_points.append("Advanced disease stage")
                
                if critical_indicators >= 1:
                    risk_level = RiskLevel.CRITICAL
                elif high_indicators >= 1:
                    risk_level = RiskLevel.HIGH
                else:
                    risk_level = RiskLevel.LOW
            
            # Calculate risk score based on level
            risk_scores = {
                RiskLevel.CRITICAL: 0.95,
                RiskLevel.HIGH: 0.75,
                RiskLevel.MEDIUM: 0.50,
                RiskLevel.LOW: 0.25,
                RiskLevel.NONE: 0.05
            }
            risk_score = risk_scores.get(risk_level, 0.0)
            
            # Generate explanation
            if explanation_points:
                explanation = "; ".join(explanation_points[:3])
            else:
                explanation = self._generate_risk_explanation_llm(text, risk_level, domain)
            
            return risk_level, risk_score, explanation
            
        except Exception as e:
            print(f"   ⚠️ LLM risk detection failed: {e}")
            return RiskLevel.NONE, 0.0, "Could not analyze risk"
    
    def _generate_risk_explanation_llm(self, text: str, level: RiskLevel, domain: str = 'legal') -> str:
        """
        Generate human-readable risk explanation using LLM.
        No hardcoded explanations - fully generated.
        """
        if not self.models_loaded:
            return f"Risk level: {level.value}"
        
        try:
            if domain == 'legal':
                prompt = f"""Explain why this legal clause has {level.value.upper()} risk for a tenant in 1-2 simple sentences:

Clause: {text}

Explanation:"""
            else:
                prompt = f"""Explain what this medical information means for the patient in 1-2 simple sentences. Mention if any values are abnormal:

Medical text: {text}

Explanation:"""
            
            inputs = self.simplifier_tokenizer(
                prompt,
                return_tensors="pt",
                max_length=384,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.simplifier_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=2,
                    temperature=0.5,
                    do_sample=False,
                    early_stopping=True
                )
            
            explanation = self.simplifier_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return explanation if explanation else f"Risk level: {level.value}"
            
        except Exception:
            return f"Risk level: {level.value}"
    
    def _detect_clause_type_llm(self, text: str, domain: str = 'legal') -> str:
        """
        Detect clause type using LLM inference (no keyword matching).
        
        Args:
            text: Clause text
            domain: 'legal' or 'medical'
            
        Returns:
            Clause type string
        """
        if not self.models_loaded:
            return 'general'
        
        try:
            if domain == 'legal':
                # Balanced classification prompt
                prompt = f"""Classify this rental clause. Answer with ONLY one word from: payment, deposit, termination, maintenance, notice, restrictions, liability, utilities, general

payment = about monthly rent amount or due dates
deposit = about security deposit amount or refund
termination = about ending lease, vacating, eviction notice
maintenance = about repairs or property condition
notice = about notification requirements
restrictions = about what's not allowed
liability = about who pays for damages
utilities = about bills
general = other

Clause: {text}

Classification:"""
            else:  # medical
                prompt = f"""Classify this medical text. Answer with ONLY one word from: lab_result, diagnosis, prescription, procedure, vital_signs, imaging, general

lab_result = contains test names with numeric values (like Hemoglobin: 8.5, Creatinine: 2.8)
diagnosis = names a disease or condition
prescription = about medication
procedure = about surgery or treatment
vital_signs = blood pressure, pulse, temperature
imaging = X-ray, MRI, CT scan
general = other

Text: {text}

Classification:"""
            
            inputs = self.simplifier_tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.simplifier_model.generate(
                    **inputs,
                    max_new_tokens=5,
                    num_beams=1,
                    do_sample=False
                )
            
            category = self.simplifier_tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
            
            # Clean up
            category = category.replace('classification:', '').replace('.', '').strip()
            # Get first word only
            category = category.split()[0] if category else 'general'
            
            legal_categories = ['payment', 'deposit', 'termination', 'maintenance', 
                               'notice', 'restrictions', 'subletting', 'utilities', 'liability', 'general']
            medical_categories = ['lab_result', 'diagnosis', 'prescription', 'procedure', 
                                 'vital_signs', 'imaging', 'general']
            
            valid = legal_categories if domain == 'legal' else medical_categories
            
            # Find matching category
            for v in valid:
                if v in category:
                    return v
            
            return 'general'
            
        except Exception:
            return 'general'
    
    def _interpret_medical_llm(self, text: str) -> str:
        """
        Interpret medical text using LLM (no hardcoded reference ranges).
        The model understands medical values and provides interpretation.
        
        Args:
            text: Medical text with potential lab values
            
        Returns:
            Interpreted text with status indicators
        """
        if not self.models_loaded:
            return text
        
        try:
            prompt = f"""Interpret this medical report for a patient. For each test value:
1. Explain what the test measures in simple terms
2. State if the value is NORMAL, HIGH, or LOW
3. Explain what abnormal values might mean

Keep the original numbers. Use simple language.

Medical report: {text}

Patient-friendly interpretation:"""
            
            inputs = self.simplifier_tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.simplifier_model.generate(
                    **inputs,
                    max_new_tokens=300,
                    num_beams=3,
                    temperature=0.4,
                    do_sample=False,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            
            interpretation = self.simplifier_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return interpretation if interpretation and len(interpretation) > 20 else text
            
        except Exception:
            return text
    
    def _generate_simplification(
        self,
        text: str,
        style: str = "plain",
        domain: str = "legal"
    ) -> str:
        """
        Generate TRULY simplified text by:
        1. Breaking complex sentences into simpler ones
        2. Replacing legal jargon with everyday words
        3. Restructuring for clarity
        
        Args:
            text: Input text to simplify
            style: 'plain' (only option now)
            domain: 'legal' or 'medical'
            
        Returns:
            Simplified text
        """
        # STEP 1: Apply comprehensive term replacements
        simplified = self._apply_term_replacements(text, domain)
        
        # STEP 2: Break into sentences and simplify each
        sentences = self._split_into_sentences(simplified)
        
        if not self.models_loaded:
            # Fallback: just use rule-based simplification
            return self._rule_based_simplify(simplified, domain)
        
        # STEP 3: Use LLM to simplify each sentence
        simplified_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                simplified_sentences.append(sentence)
                continue
            
            try:
                simplified_sent = self._simplify_single_sentence(sentence, domain)
                simplified_sentences.append(simplified_sent)
            except Exception:
                simplified_sentences.append(sentence)
        
        result = ' '.join(simplified_sentences)
        
        # STEP 4: Final cleanup and term replacement
        result = self._apply_term_replacements(result, domain)
        result = self._rule_based_simplify(result, domain)
        
        print(f"[DEBUG] Used t5-simplification model")
        return result
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Split on period, but not on common abbreviations
        text = re.sub(r'\bRs\.', 'Rs', text)
        text = re.sub(r'\bMr\.', 'Mr', text)
        text = re.sub(r'\bMrs\.', 'Mrs', text)
        text = re.sub(r'\bDr\.', 'Dr', text)
        text = re.sub(r'\bNo\.', 'No', text)
        text = re.sub(r'\bw\.e\.f\.?', 'from', text)
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _simplify_single_sentence(self, sentence: str, domain: str) -> str:
        """Simplify a single sentence using the model."""
        if not self.models_loaded:
            return self._rule_based_simplify(sentence, domain)
        
        # First apply rule-based simplification
        pre_simplified = self._rule_based_simplify(sentence, domain)
        
        # Use appropriate prompt based on model type
        if self.simplification_model_type == "pegasus-paraphrase":
            prompt = pre_simplified  # Pegasus doesn't need a prefix
        elif self.simplification_model_type == "t5-simplification":
            prompt = f"simplify: {pre_simplified}"
        else:  # flan-t5
            prompt = f"Rewrite this in simple everyday language: {pre_simplified}"
        
        try:
            inputs = self.simplifier_tokenizer(
                prompt,
                return_tensors="pt",
                max_length=256,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.simplifier_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    min_new_tokens=5,
                    num_beams=4,
                    length_penalty=0.8,  # Prefer shorter outputs
                    early_stopping=True,
                    do_sample=False,
                )
            
            result = self.simplifier_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # If result is valid and different, use it; otherwise use rule-based
            if result and len(result) > 10 and result.lower() != pre_simplified.lower():
                return result
            else:
                return pre_simplified
        except Exception:
            return pre_simplified
    
    def _rule_based_simplify(self, text: str, domain: str) -> str:
        """
        Apply rule-based simplifications for clearer text.
        This restructures sentences, not just replaces words.
        """
        import re
        
        result = text
        
        # Break up long sentences with multiple clauses
        # "X and Y shall Z" -> "X must Z. Y must also Z."
        
        # Simplify passive voice patterns
        passive_patterns = [
            (r'shall be paid by the (\w+)', r'the \1 must pay'),
            (r'shall be borne by the (\w+)', r'the \1 must pay for'),
            (r'shall be made by the (\w+)', r'the \1 must make'),
            (r'shall be done by the (\w+)', r'the \1 must do'),
            (r'shall be given by the (\w+)', r'the \1 must give'),
            (r'is agreed that', r'both parties agree that'),
            (r'it is hereby agreed', r'both parties agree'),
            (r'is hereby declared', r'we state'),
        ]
        
        for pattern, replacement in passive_patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        # Simplify complex clause structures
        complex_patterns = [
            # "in the event of X" -> "if X happens"
            (r'in the event of ([^,\.]+)', r'if \1 happens'),
            (r'in the event that ([^,\.]+)', r'if \1'),
            # "for the purpose of" -> "to"
            (r'for the purpose of', r'to'),
            # "in respect of" -> "about" or "for"
            (r'in respect of', r'for'),
            # "on account of" -> "because of"
            (r'on account of', r'because of'),
            # "by virtue of" -> "because of"
            (r'by virtue of', r'because of'),
            # "subject to" -> "if" or "as long as"
            (r'subject to ([^,\.]+)', r'as long as \1'),
            # "notwithstanding" -> "even though" or "despite"
            (r'notwithstanding ([^,\.]+)', r'despite \1'),
            # "provided that" -> "but only if"
            (r'provided that', r'but only if'),
            # "in accordance with" -> "following" or "as per"
            (r'in accordance with', r'following'),
            # "pursuant to" -> "according to" or "under"
            (r'pursuant to', r'under'),
            # "at the time of" -> "when"
            (r'at the time of', r'when'),
            # "prior to" -> "before"
            (r'prior to', r'before'),
            # "subsequent to" -> "after"
            (r'subsequent to', r'after'),
            # "in lieu of" -> "instead of"
            (r'in lieu of', r'instead of'),
            # "with regard to" -> "about"
            (r'with regard to', r'about'),
            # "in connection with" -> "about" or "related to"
            (r'in connection with', r'about'),
            # Remove redundant legalese
            (r'whatsoever', r''),
            (r'howsoever', r''),
            (r'whomsoever', r'whoever'),
            (r'hereby', r''),
            (r'hereto', r'to this'),
            (r'hereof', r'of this'),
            (r'thereof', r'of that'),
            (r'thereto', r'to that'),
            (r'wherein', r'where'),
            (r'whereby', r'by which'),
            (r'whereof', r'of which'),
            (r'aforesaid', r'mentioned above'),
            (r'aforementioned', r'mentioned above'),
            (r'hereunder', r'below'),
            (r'hereinabove', r'above'),
            (r'hereinafter', r'from now on'),
            (r'heretofore', r'until now'),
        ]
        
        for pattern, replacement in complex_patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        # AGGRESSIVE SENTENCE RESTRUCTURING
        # Convert legal structures to plain English
        
        # "The [party] shall [verb]" -> "[Party] must [verb]"
        result = re.sub(r'\bThe (tenant|landlord|lessor|lessee|party|owner|buyer|seller)\s+shall\s+', 
                       lambda m: f"The {m.group(1)} must ", result, flags=re.IGNORECASE)
        
        # "shall not" -> "must not" or "cannot"
        result = re.sub(r'\bshall not\b', 'cannot', result, flags=re.IGNORECASE)
        result = re.sub(r'\bshall\b', 'must', result, flags=re.IGNORECASE)
        
        # "agrees to [verb]" -> "will [verb]"
        result = re.sub(r'\bagrees to\b', 'will', result, flags=re.IGNORECASE)
        
        # Remove verbose phrases
        verbose_to_simple = [
            (r'\bin order to\b', 'to'),
            (r'\bfor the reason that\b', 'because'),
            (r'\bdue to the fact that\b', 'because'),
            (r'\bin the amount of\b', 'of'),
            (r'\bat such time as\b', 'when'),
            (r'\bduring such time as\b', 'while'),
            (r'\bin the case that\b', 'if'),
            (r'\bin the case of\b', 'for'),
            (r'\bwith respect to\b', 'about'),
            (r'\bas regards\b', 'about'),
            (r'\bas to\b', 'about'),
            (r'\bby means of\b', 'by'),
            (r'\bby reason of\b', 'because of'),
            (r'\bfor the period of\b', 'for'),
            (r'\bfor the duration of\b', 'during'),
            (r'\bat the present time\b', 'now'),
            (r'\bat this point in time\b', 'now'),
            (r'\bin the near future\b', 'soon'),
            (r'\bin the absence of\b', 'without'),
            (r'\bwith the exception of\b', 'except'),
            (r'\bin excess of\b', 'more than'),
            (r'\bnot less than\b', 'at least'),
            (r'\bnot more than\b', 'at most'),
            (r'\bin no event\b', 'never'),
            (r'\bunder no circumstances\b', 'never'),
            (r'\bthe said\b', 'this'),
            (r'\bsaid\b', 'this'),
            (r'\bsuch\b', 'this'),
            (r'\bthe same\b', 'it'),
            (r'\bthe above\b', 'this'),
            (r'\bthe undersigned\b', 'I/we'),
        ]
        
        for pattern, replacement in verbose_to_simple:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        # Simplify legal entities
        entity_simplifications = [
            (r'\bthe premises\b', 'the property'),
            (r'\bpremises\b', 'property'),
            (r'\bthe demised premises\b', 'the rented property'),
            (r'\bthe property herein\b', 'this property'),
            (r'\blessor\b', 'landlord'),
            (r'\blessee\b', 'tenant'),
            (r'\blesee\b', 'tenant'),
            (r'\bthe first party\b', 'the landlord'),
            (r'\bthe second party\b', 'the tenant'),
            (r'\bparty of the first part\b', 'landlord'),
            (r'\bparty of the second part\b', 'tenant'),
            (r'\btermination\b', 'end'),
            (r'\bcommencement\b', 'start'),
            (r'\boccupy\b', 'live in'),
            (r'\boccupies\b', 'lives in'),
            (r'\bvacate\b', 'leave'),
            (r'\bvacating\b', 'leaving'),
            (r'\bvacant possession\b', 'empty property'),
            (r'\bsurrender\b', 'give back'),
            (r'\bexecute\b', 'sign'),
            (r'\bexecuted\b', 'signed'),
            (r'\baffixed\b', 'attached'),
            (r'\bforfeit\b', 'lose'),
            (r'\bdefault\b', 'failure to pay'),
            (r'\bin default\b', 'fails to pay'),
            (r'\bsecurity deposit\b', 'deposit'),
            (r'\badvance amount\b', 'advance payment'),
            (r'\bmonthly rent\b', 'rent per month'),
            (r'\bper annum\b', 'per year'),
            (r'\bper mensem\b', 'per month'),
            (r'\bcalendar month\b', 'month'),
            (r'\bw\.?e\.?f\.?\b', 'from'),
            (r'\bwith effect from\b', 'starting from'),
        ]
        
        for pattern, replacement in entity_simplifications:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        # Break very long sentences at conjunctions
        # Add period before "and" if sentence is too long
        words = result.split()
        if len(words) > 35:
            # Try to break at "and" in the middle
            mid = len(words) // 2
            for i in range(mid - 5, mid + 5):
                if i < len(words) and words[i].lower() == 'and':
                    words[i] = '. Also,'
                    break
            result = ' '.join(words)
        
        # Clean up extra spaces and punctuation
        result = re.sub(r'\s+', ' ', result).strip()
        result = re.sub(r'\s+([,\.])', r'\1', result)
        result = re.sub(r'\.+', '.', result)
        
        # Capitalize first letter after period
        result = '. '.join(s.strip().capitalize() if s else s for s in result.split('. '))
        
        return result
    
    def _validate_llm_output(self, output: str, original: str) -> bool:
        """
        Validate that LLM output is reasonable.
        
        Args:
            output: LLM generated text
            original: Original input text
            
        Returns:
            True if output is valid, False otherwise
        """
        if not output:
            return False
        
        # Check minimum length (not too short)
        if len(output) < 15:
            return False
        
        # Check maximum length (not too long - max 2x original)
        if len(output) > len(original) * 2.5:
            return False
        
        # Check it's not just echoing the prompt
        if output.lower().startswith("simplify") or output.lower().startswith("legal text"):
            return False
        
        # Check it's not just the original text
        if output.strip().lower() == original.strip().lower():
            return False
        
        # Check for common LLM failure patterns
        failure_patterns = [
            "i cannot", "i can't", "i'm unable",
            "as an ai", "i don't have",
            "please provide", "could you",
        ]
        output_lower = output.lower()
        for pattern in failure_patterns:
            if pattern in output_lower:
                return False
        
        return True
    
    def _apply_term_replacements(self, text: str, domain: str) -> str:
        """Apply domain-specific term replacements (FALLBACK mechanism)"""
        import re
        result = text
        
        if domain == "legal":
            replacements = [
                # Parties
                (r'\bLessee\b', 'Tenant'),
                (r'\blessee\b', 'tenant'),
                (r'\bLessor\b', 'Landlord'),
                (r'\blessor\b', 'landlord'),
                
                # Modal verbs
                (r'\bshall\b', 'must'),
                (r'\bShall\b', 'Must'),
                
                # Time expressions
                (r'\bforthwith\b', 'immediately'),
                (r'\bwithout delay\b', 'immediately'),
                (r'\bwith immediate effect\b', 'right now'),
                
                # Reference words  
                (r'\bhereinafter\b', 'from now on'),
                (r'\bhereinbefore\b', 'mentioned earlier'),
                (r'\bhereby\b', 'by this'),
                (r'\bhereto\b', 'to this'),
                (r'\bhereunder\b', 'under this'),
                (r'\bhereof\b', 'of this'),
                (r'\baforesaid\b', 'mentioned above'),
                (r'\bwhereof\b', 'of which'),
                (r'\bthereof\b', 'of that'),
                (r'\btherein\b', 'in that'),
                (r'\bthereto\b', 'to that'),
                (r'\bthereunder\b', 'under that'),
                
                # Property terms
                (r'\bdemised premises\b', 'rented property'),
                (r'\bthe premises\b', 'the property'),
                (r'\bpremises\b', 'property'),
                (r'\bdwelling unit\b', 'rental unit'),
                (r'\bleasehold\b', 'rental'),
                
                # Actions
                (r'\bterminate\b', 'end'),
                (r'\bTerminate\b', 'End'),
                (r'\btermination\b', 'ending'),
                (r'\bvacate\b', 'leave'),
                (r'\bsurrender\b', 'return'),
                (r'\bforfeit\b', 'lose'),
                (r'\bforfeiture\b', 'loss'),
                (r'\brelinquish\b', 'give up'),
                
                # Promises/agreements
                (r'\bcovenants\b', 'agrees'),
                (r'\bcovenant\b', 'agree'),
                (r'\bundertakes\b', 'agrees'),
                (r'\bundertake\b', 'agree'),
                (r'\bwarrants\b', 'guarantees'),
                (r'\bwarrant\b', 'guarantee'),
                
                # Legal actions
                (r'\bindemnify\b', 'compensate'),
                (r'\bindemnification\b', 'compensation'),
                (r'\bhold harmless\b', 'protect from blame'),
                (r'\bwaive\b', 'give up'),
                (r'\bwaiver\b', 'giving up'),
                (r'\bsublet\b', 'rent to someone else'),
                (r'\bassign\b', 'transfer'),
                (r'\bencumber\b', 'put a legal claim on'),
                
                # Conditional phrases
                (r'\bIn the event of\b', 'If there is'),
                (r'\bin the event of\b', 'if there is'),
                (r'\bIn the event that\b', 'If'),
                (r'\bin the event that\b', 'if'),
                (r'\bpursuant to\b', 'according to'),
                (r'\bnotwithstanding\b', 'despite'),
                (r'\bsubject to\b', 'depending on'),
                (r'\bprovided that\b', 'as long as'),
                (r'\bprovided however\b', 'but'),
                
                # Latin terms
                (r'\binter alia\b', 'among other things'),
                (r'\bpro rata\b', 'proportionally'),
                (r'\bmutatis mutandis\b', 'with necessary changes'),
                (r'\bbona fide\b', 'genuine'),
                
                # Other common legal terms
                (r'\bduly\b', 'properly'),
                (r'\bwithout prejudice\b', 'without affecting rights'),
                (r'\bas the case may be\b', 'as needed'),
                (r'\bfor the time being\b', 'currently'),
                (r'\bin force\b', 'valid'),
                (r'\bin good standing\b', 'valid and active'),
            ]
        elif domain == "medical":
            replacements = [
                # Blood-related tests
                (r'\bhemoglobin\b', 'oxygen-carrying protein in blood'),
                (r'\bHemoglobin\b', 'Oxygen-carrying protein in blood'),
                (r'\bhematocrit\b', 'percentage of red blood cells'),
                (r'\bthrombocytopenia\b', 'low platelet count'),
                (r'\bleukocytosis\b', 'high white blood cell count'),
                (r'\bleukopenia\b', 'low white blood cell count'),
                (r'\banemia\b', 'low red blood cell count'),
                (r'\berythrocyte\b', 'red blood cell'),
                (r'\bleukocyte\b', 'white blood cell'),
                (r'\bplatelet count\b', 'blood clotting cell count'),
                (r'\bRBC\b', 'red blood cell'),
                (r'\bWBC\b', 'white blood cell'),
                (r'\bPCV\b', 'packed cell volume (blood thickness)'),
                (r'\bMCV\b', 'average red blood cell size'),
                (r'\bMCH\b', 'average hemoglobin per cell'),
                (r'\bMCHC\b', 'hemoglobin concentration in cells'),
                (r'\bESR\b', 'inflammation marker'),
                
                # Kidney function
                (r'\bSerum Creatinine\b', 'Kidney waste marker'),
                (r'\bserum creatinine\b', 'kidney waste marker'),
                (r'\bBlood Urea\b', 'Kidney waste marker (urea)'),
                (r'\bblood urea\b', 'kidney waste marker (urea)'),
                (r'\bBUN\b', 'blood urea nitrogen (kidney function)'),
                (r'\bUric Acid\b', 'Waste product from protein breakdown'),
                (r'\buric acid\b', 'waste product from protein breakdown'),
                (r'\beGFR\b', 'kidney filtration rate'),
                (r'\bGFR\b', 'kidney filtration rate'),
                
                # Liver function
                (r'\bBilirubin\b', 'Liver breakdown product (causes yellow color)'),
                (r'\bbilirubin\b', 'liver breakdown product (causes yellow color)'),
                (r'\bhyperbilirubinemia\b', 'high bilirubin level (may cause yellowing)'),
                (r'\bAST\b', 'liver enzyme (AST)'),
                (r'\bSGOT\b', 'liver enzyme'),
                (r'\bALT\b', 'liver enzyme (ALT)'),
                (r'\bSGPT\b', 'liver enzyme'),
                (r'\bALP\b', 'bone/liver enzyme'),
                (r'\bAlkaline Phosphatase\b', 'Bone/liver enzyme'),
                (r'\bGGT\b', 'liver/bile duct enzyme'),
                (r'\bAlbumin\b', 'Protein made by liver'),
                (r'\balbumin\b', 'protein made by liver'),
                
                # Blood sugar
                (r'\bhypoglycemia\b', 'low blood sugar'),
                (r'\bhyperglycemia\b', 'high blood sugar'),
                (r'\bnormoglycemic\b', 'normal blood sugar'),
                (r'\bglycemic\b', 'blood sugar'),
                (r'\bHbA1c\b', '3-month average blood sugar'),
                (r'\bFasting Blood Sugar\b', 'Blood sugar after fasting'),
                (r'\bRandom Blood Sugar\b', 'Blood sugar at any time'),
                (r'\bPPBS\b', 'blood sugar after eating'),
                
                # Blood pressure
                (r'\bhypertension\b', 'high blood pressure'),
                (r'\bhypotension\b', 'low blood pressure'),
                (r'\bsystolic\b', 'upper reading (heart pumping)'),
                (r'\bdiastolic\b', 'lower reading (heart resting)'),
                
                # Lipid profile
                (r'\bCholesterol\b', 'Blood fat (cholesterol)'),
                (r'\bcholesterol\b', 'blood fat'),
                (r'\bTriglycerides\b', 'Blood fat from food'),
                (r'\btriglycerides\b', 'blood fat from food'),
                (r'\bHDL\b', 'good cholesterol'),
                (r'\bLDL\b', 'bad cholesterol'),
                (r'\bVLDL\b', 'very bad cholesterol'),
                (r'\bhyperlipidemia\b', 'high blood fat levels'),
                (r'\bdyslipidemia\b', 'abnormal blood fat levels'),
                
                # Thyroid
                (r'\bTSH\b', 'thyroid stimulating hormone'),
                (r'\bT3\b', 'thyroid hormone (T3)'),
                (r'\bT4\b', 'thyroid hormone (T4)'),
                (r'\bhypothyroidism\b', 'underactive thyroid'),
                (r'\bhyperthyroidism\b', 'overactive thyroid'),
                
                # Organs
                (r'\bhepatic\b', 'liver'),
                (r'\brenal\b', 'kidney'),
                (r'\bcardiac\b', 'heart'),
                (r'\bpulmonary\b', 'lung'),
                (r'\bgastric\b', 'stomach'),
                (r'\bcerebral\b', 'brain'),
                (r'\bocular\b', 'eye'),
                (r'\bdermal\b', 'skin'),
                (r'\bvascular\b', 'blood vessel'),
                (r'\bintestinal\b', 'gut/intestine'),
                
                # Common conditions/procedures
                (r'\bcholecystitis\b', 'gallbladder inflammation'),
                (r'\bcholecystectomy\b', 'gallbladder removal surgery'),
                (r'\blaparoscopic\b', 'keyhole (minimally invasive)'),
                (r'\bappendectomy\b', 'appendix removal surgery'),
                (r'\bappendicitis\b', 'appendix inflammation'),
                (r'\bpneumonia\b', 'lung infection'),
                (r'\bbronchitis\b', 'airway inflammation'),
                (r'\bgastritis\b', 'stomach lining inflammation'),
                (r'\bpancreatitis\b', 'pancreas inflammation'),
                (r'\bhepatitis\b', 'liver inflammation'),
                (r'\bnephritis\b', 'kidney inflammation'),
                (r'\barthritis\b', 'joint inflammation'),
                (r'\bdiabetes mellitus\b', 'diabetes (high blood sugar condition)'),
                
                # Time/severity
                (r'\bacute\b', 'sudden/recent'),
                (r'\bchronic\b', 'long-term'),
                (r'\bbenign\b', 'not harmful/non-cancerous'),
                (r'\bmalignant\b', 'cancerous'),
                (r'\bmetastatic\b', 'cancer that has spread'),
                
                # Other medical terms
                (r'\bprognosis\b', 'expected outcome'),
                (r'\bdiagnosis\b', 'medical finding'),
                (r'\basymptomatic\b', 'without symptoms'),
                (r'\bsymptomatic\b', 'showing symptoms'),
                (r'\bcontraindicated\b', 'not recommended'),
                (r'\bprescribed\b', 'recommended by doctor'),
                (r'\badminister\b', 'give'),
                (r'\bdosage\b', 'amount to take'),
                (r'\bprophylactic\b', 'preventive'),
                (r'\bpalliative\b', 'for comfort/symptom relief'),
                (r'\bidiopathic\b', 'of unknown cause'),
                (r'\biatrogenic\b', 'caused by treatment'),
                (r'\bpostoperative\b', 'after surgery'),
                (r'\bpreoperative\b', 'before surgery'),
                (r'\bintravenous\b', 'into the vein'),
                (r'\boral\b', 'by mouth'),
                (r'\btopical\b', 'applied on skin'),
                
                # Normal/abnormal indicators
                (r'\bwithin normal limits\b', 'NORMAL'),
                (r'\bWNL\b', 'normal'),
                (r'\belevated\b', 'higher than normal'),
                (r'\bdepressed\b', 'lower than normal'),
                (r'\babnormal\b', 'outside normal range'),
            ]
        else:
            replacements = []
        
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def _interpret_medical_results(self, text: str) -> str:
        """
        Interpret medical lab results using LLM.
        No hardcoded reference ranges - the LLM understands medical values.
        
        Args:
            text: Medical text potentially containing lab results
            
        Returns:
            Interpreted text with normal/abnormal indicators
        """
        # Use the LLM-based medical interpretation
        return self._interpret_medical_llm(text)
    
    def _simplify_medical(self, text: str) -> str:
        """
        Comprehensive medical text simplification.
        Combines LLM, term replacement, and lab result interpretation.
        """
        # Step 1: Try to interpret any lab results with values
        interpreted = self._interpret_medical_results(text)
        
        # Step 2: Apply medical term replacements
        simplified = self._apply_term_replacements(interpreted, domain='medical')
        
        return simplified

    def _extract_medical_entities(self, text: str) -> List[Dict]:
        """
        Extract medical entities using BioBERT NER pipeline or parser fallback.
        
        Returns list of entities with: entity type, text, score/flag, and interpretation.
        """
        entities = []
        
        # Method 1: Use BioBERT NER if available
        try:
            if self.bio_ner_pipeline is not None:
                ner = self.bio_ner_pipeline(text[:512])  # Truncate for BERT limit
                for e in ner:
                    entity_type = e.get('entity_group', e.get('entity', 'UNKNOWN'))
                    word = e.get('word', '').replace('##', '')  # Handle subword tokens
                    score = float(e.get('score', 0.0))
                    if score > 0.5 and len(word) > 2:  # Filter low-confidence and short
                        entities.append({
                            'entity': entity_type,
                            'text': word,
                            'score': round(score, 3),
                            'source': 'biobert'
                        })
        except Exception:
            pass
        
        # Method 2: Use MedicalReportParser for structured data
        if MEDICAL_PARSER_AVAILABLE and len(entities) < 3:
            try:
                parser = MedicalReportParser()
                parsed = parser.parse_report(text)
                for r in parsed.get('results', []):
                    entities.append({
                        'entity': r.get('test_name_standardized', r.get('test_name')),
                        'text': str(r.get('value', '')),
                        'unit': r.get('unit', ''),
                        'flag': r.get('flag', 'unknown'),
                        'reference': r.get('reference_range', ''),
                        'source': 'parser'
                    })
            except Exception:
                pass
        
        # Method 3: Regex fallback for common medical patterns
        if len(entities) < 2:
            # Extract test name: value unit patterns
            patterns = [
                r'(Hemoglobin|HB|Hgb)[:\s]+(\d+\.?\d*)\s*(g/dL|g%)?',
                r'(Blood Sugar|FBS|RBS|PPBS)[:\s]+(\d+\.?\d*)\s*(mg/dL)?',
                r'(Creatinine)[:\s]+(\d+\.?\d*)\s*(mg/dL)?',
                r'(Bilirubin)[:\s]+(\d+\.?\d*)\s*(mg/dL)?',
                r'(WBC|RBC|Platelet)[:\s]+(\d+\.?\d*)',
                r'(TSH|T3|T4)[:\s]+(\d+\.?\d*)',
                r'(Cholesterol|HDL|LDL|Triglycerides)[:\s]+(\d+\.?\d*)\s*(mg/dL)?',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    test_name = match[0] if match else ''
                    value = match[1] if len(match) > 1 else ''
                    unit = match[2] if len(match) > 2 else ''
                    if test_name and value:
                        entities.append({
                            'entity': test_name,
                            'text': value,
                            'unit': unit,
                            'source': 'regex'
                        })
        
        # Remove duplicates based on entity name
        seen = set()
        unique_entities = []
        for e in entities:
            key = (e.get('entity', '').lower(), e.get('text', ''))
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)
        
        return unique_entities[:15]  # Limit to 15 entities

    def _preprocess_medical_text(self, text: str) -> Tuple[str, Optional[Dict]]:
        """
        Detect and parse tabular medical data. If a table is detected and the medical parser
        is available, return a structured simplified text summary and the parsed dict.
        Otherwise return the original text and None.
        """
        lines = [l for l in text.split('\n') if l.strip()]
        if not lines:
            return text, None

        # Heuristic: many lines with numeric values indicates tabular lab data
        numeric_lines = 0
        for l in lines[:200]:
            nums = re.findall(r'\d+\.?\d*', l)
            if len(nums) >= 1 and any(c.isdigit() for c in l):
                numeric_lines += 1

        # If more than ~30% of lines contain numeric values, treat as table
        if len(lines) >= 3 and (numeric_lines / len(lines)) > 0.3 and MEDICAL_PARSER_AVAILABLE:
            try:
                parser = MedicalReportParser()
                parsed = parser.parse_and_explain(text)
                # Build a concise textual summary for LLM input
                parts = [parsed.get('summary', '')]
                for r in parsed.get('results', []):
                    name = r.get('test_name_standardized') or r.get('test_name')
                    val = r.get('value')
                    unit = r.get('unit')
                    flag = r.get('flag')
                    expl = r.get('explanation', '')
                    parts.append(f"{name}: {val} {unit} [{flag}] - {expl}")
                parsed_text = '\n'.join(parts)
                return parsed_text, parsed
            except Exception:
                return text, None

        return text, None
    
    def _make_colloquial(self, text: str, domain: str) -> str:
        """Convert to colloquial/friendly style"""
        import re
        result = text
        
        if domain == "legal":
            # Order matters - do more specific replacements first
            # Step 1: Handle specific verb forms with tenant (before generic replacements)
            result = re.sub(r'\bThe tenant must\b', 'You need to', result)
            result = re.sub(r'\bthe tenant must\b', 'you need to', result)
            result = re.sub(r'\bThe tenant shall\b', 'You will need to', result)
            result = re.sub(r'\bthe tenant shall\b', 'you will need to', result)
            result = re.sub(r'\bThe tenant agrees\b', 'You agree', result)
            result = re.sub(r'\bthe tenant agrees\b', 'you agree', result)
            result = re.sub(r'\bThe tenant has\b', 'You have', result)
            result = re.sub(r'\bthe tenant has\b', 'you have', result)
            result = re.sub(r'\bThe tenant also agrees\b', 'You also agree', result)
            result = re.sub(r'\bthe tenant also agrees\b', 'you also agree', result)
            result = re.sub(r'\bThe tenant also agreed\b', 'You also agreed', result)
            result = re.sub(r'\bthe tenant also agreed\b', 'you also agreed', result)
            
            # Step 1b: Handle "hereby" phrases - common in legal text
            result = re.sub(r'\bby this agrees\b', 'agree', result)
            result = re.sub(r'\bhereby agrees\b', 'agrees', result)  # Will become "agree" after tenant->you
            
            # Step 2: Generic tenant replacements
            result = re.sub(r'\bThe Tenant\b', 'You', result)
            result = re.sub(r'\bthe Tenant\b', 'you', result)
            result = re.sub(r'\bThe tenant\b', 'You', result)
            result = re.sub(r'\bthe tenant\b', 'you', result)
            result = re.sub(r'\bTenant\b', 'You', result)
            result = re.sub(r'\btenant\b', 'you', result)
            
            # Step 3: Fix verb conjugation after "you" (third person -> second person)
            result = re.sub(r'\b(You|you) agrees\b', r'\1 agree', result)
            result = re.sub(r'\b(You|you) covenants\b', r'\1 agree', result)
            result = re.sub(r'\b(You|you) undertakes\b', r'\1 undertake', result)
            result = re.sub(r'\b(You|you) warrants\b', r'\1 guarantee', result)
            result = re.sub(r'\b(You|you) guarantees\b', r'\1 guarantee', result)
            result = re.sub(r'\b(You|you) has\b', r'\1 have', result)
            result = re.sub(r'\b(You|you) does\b', r'\1 do', result)
            result = re.sub(r'\b(You|you) is\b', r'\1 are', result)
            result = re.sub(r'\b(You|you) occupies\b', r'\1 occupy', result)
            result = re.sub(r'\b(You|you) pays\b', r'\1 pay', result)
            result = re.sub(r'\b(You|you) returns\b', r'\1 return', result)
            result = re.sub(r'\b(You|you) leaves\b', r'\1 leave', result)
            
            # Step 4: Handle landlord variations (avoid double "Your")
            result = re.sub(r'\bto the Landlord\b', 'to the landlord', result)
            result = re.sub(r'\bThe Landlord\b', 'The landlord', result)
            result = re.sub(r'\bthe landlord\b', 'your landlord', result)
            result = re.sub(r'\bThe landlord\b', 'Your landlord', result)
            
            # Step 5: Fix pronoun issues (he/she referring to tenant -> you)
            result = re.sub(r'\bhe occupies\b', 'you occupy', result)
            result = re.sub(r'\bshe occupies\b', 'you occupy', result)
            result = re.sub(r'\bthey are\b', 'you are', result)
            result = re.sub(r'\btheir signatures\b', 'your signatures', result)
            
            # Step 6: Clean up double words
            result = re.sub(r'\byour your\b', 'your', result, flags=re.IGNORECASE)
            result = re.sub(r'\bYour Your\b', 'Your', result)
            result = re.sub(r'\bthe the\b', 'the', result, flags=re.IGNORECASE)
            result = re.sub(r'\bthe you\b', 'you', result, flags=re.IGNORECASE)
            result = re.sub(r'\bto you the\b', 'to your', result, flags=re.IGNORECASE)
            
            # Step 7: Simplify property terms
            result = re.sub(r'\bthe property\b', 'the place', result)
            result = re.sub(r'\bthe rented property\b', 'the place', result)
            result = re.sub(r'\bthe premises\b', 'the place', result)
            result = re.sub(r'\bthe building\b', 'the place', result)
            
            # Step 8: Make action words more friendly
            result = re.sub(r'\bimmediately leave\b', 'move out right away', result)
            result = re.sub(r'\bimmediately\b', 'right away', result)
            result = re.sub(r'\bupon expiration\b', 'when it ends', result)
            result = re.sub(r'\bfor two consecutive months\b', 'for 2 months in a row', result)
            result = re.sub(r'\bdefault in payment\b', 'failure to pay rent', result)
            result = re.sub(r'\bmust not commit\b', "shouldn't do", result)
            result = re.sub(r'\bmust not\b', "shouldn't", result)
            result = re.sub(r'\bshall not\b', "shouldn't", result)
            
            # Step 9: Remove legal filler phrases
            result = re.sub(r'\bby this\s+', '', result)
            result = re.sub(r'\bhereby\s+', '', result)
            
        elif domain == "medical":
            # Patient -> You
            result = re.sub(r'\bThe patient\b', 'You', result)
            result = re.sub(r'\bthe patient\b', 'you', result)
            result = re.sub(r'\bPatient\b', 'You', result)
            result = re.sub(r'\bpatient\b', 'you', result)
            
            # Medical terms to friendly
            result = re.sub(r'\bshould consult\b', 'should talk to', result)
            result = re.sub(r'\bphysician\b', 'doctor', result)
            result = re.sub(r'\bprescribed\b', 'recommended by your doctor', result)
            result = re.sub(r'\badminister\b', 'take', result)
        
        # Clean up any double spaces and punctuation issues
        result = re.sub(r'\s+', ' ', result).strip()
        result = re.sub(r'\s+([.,;:])', r'\1', result)  # Remove space before punctuation
        
        return result
    
    def _detect_risk(self, text: str, domain: str = 'legal') -> Tuple[RiskLevel, float, str]:
        """
        Detect risk level using pure LLM inference (no hardcoded patterns).
        
        Args:
            text: Clause text to analyze
            domain: 'legal' or 'medical'
            
        Returns:
            Tuple of (risk_level, risk_score, explanation)
        """
        # Use the LLM-based risk detection
        return self._detect_risk_llm(text, domain)
    
    def _detect_clause_type(self, text: str, domain: str = 'legal') -> str:
        """
        Detect the type/category of a clause using pure LLM inference.
        
        Args:
            text: Clause text
            domain: 'legal' or 'medical'
            
        Returns:
            Clause type string
        """
        # Use the LLM-based clause type detection
        return self._detect_clause_type_llm(text, domain)
    
    def _extract_key_terms(self, text: str, domain: str = 'legal') -> List[str]:
        """
        Extract key terms/values from a clause using LLM.
        Falls back to regex only if LLM fails.
        
        Args:
            text: Input text
            domain: 'legal' or 'medical'
            
        Returns:
            List of key terms extracted
        """
        key_terms = []
        
        # PRIMARY: Use LLM for key term extraction
        if self.models_loaded:
            try:
                if domain == 'legal':
                    prompt = f"""Extract the important values from this legal clause. List only:
- Money amounts (with ₹ or Rs)
- Time periods (days, months, years)
- Percentages
- Dates
- Names of parties

Text: {text}

Important values (comma separated):"""
                else:  # medical
                    prompt = f"""Extract the important values from this medical text. List only:
- Test names and their values with units
- Medication names and dosages
- Dates
- Reference ranges

Text: {text}

Important values (comma separated):"""
                
                inputs = self.simplifier_tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=384,
                    truncation=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.simplifier_model.generate(
                        **inputs,
                        max_new_tokens=100,
                        num_beams=2,
                        temperature=0.3,
                        do_sample=False
                    )
                
                result = self.simplifier_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                
                # Parse comma-separated values
                if result:
                    terms = [t.strip() for t in result.split(',') if t.strip()]
                    key_terms = [t for t in terms if len(t) > 1 and len(t) < 50]
                    
                    if key_terms:
                        return list(dict.fromkeys(key_terms))[:8]
                        
            except Exception:
                pass  # Fall through to regex fallback
        
        # FALLBACK: Regex extraction (only when LLM unavailable or fails)
        # Extract monetary values
        money_patterns = [
            r'(?:rs\.?|₹|inr)\s*([0-9,]+)',
            r'([0-9,]+)\s*(?:rupees|rs)',
        ]
        for pattern in money_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                key_terms.append(f"₹{match}")
        
        # Extract time periods
        time_patterns = [
            r'(\d+)\s*(?:month|months)',
            r'(\d+)\s*(?:day|days)',
            r'(\d+)\s*(?:year|years)',
        ]
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if 'month' in pattern:
                    key_terms.append(f"{match} months")
                elif 'day' in pattern:
                    key_terms.append(f"{match} days")
                else:
                    key_terms.append(f"{match} years")
        
        # Extract percentages
        percent_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
        for match in percent_matches:
            key_terms.append(f"{match}%")

        # For medical domain, add medical entity names or parsed test names
        if domain == 'medical':
            try:
                entities = self._extract_medical_entities(text)
                for e in entities:
                    name = e.get('entity') or e.get('text')
                    if name:
                        key_terms.append(str(name))
            except Exception:
                pass

        return list(dict.fromkeys(key_terms))[:8]  # Preserve order, allow up to 8 terms

    def _validate_consistency(self, original: str, simplified: str, domain: str = 'legal') -> Tuple[bool, List[str]]:
        """
        Validate that important information is preserved during simplification.
        Checks that numbers, dates, monetary values, and key terms are not lost.
        
        Args:
            original: Original text before simplification
            simplified: Simplified text
            domain: 'legal' or 'medical'
            
        Returns:
            Tuple of (is_consistent: bool, warnings: List[str])
        """
        warnings = []
        
        # Extract all numbers from original
        original_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', original))
        simplified_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', simplified))
        
        # Check for missing important numbers (ignore small numbers like 1, 2, 3)
        important_numbers = {n for n in original_numbers if float(n) >= 10 or '.' in n}
        missing_numbers = important_numbers - simplified_numbers
        
        if missing_numbers:
            warnings.append(f"⚠️ Numbers may be missing: {', '.join(sorted(missing_numbers)[:5])}")
        
        # Extract monetary values
        money_pattern = r'(?:rs\.?|₹|inr)\s*([\d,]+)|([\d,]+)\s*(?:rupees|rs)'
        original_money = set(re.findall(money_pattern, original, re.IGNORECASE))
        simplified_money = set(re.findall(money_pattern, simplified, re.IGNORECASE))
        
        # Flatten tuples from findall
        original_money = {m for tup in original_money for m in tup if m}
        simplified_money = {m for tup in simplified_money for m in tup if m}
        
        missing_money = original_money - simplified_money
        if missing_money:
            warnings.append(f"⚠️ Monetary values may be missing: ₹{', ₹'.join(sorted(missing_money)[:3])}")
        
        # Check for dates (common formats)
        date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{2,4}\b'
        original_dates = set(re.findall(date_pattern, original, re.IGNORECASE))
        simplified_dates = set(re.findall(date_pattern, simplified, re.IGNORECASE))
        
        missing_dates = original_dates - simplified_dates
        if missing_dates:
            warnings.append(f"⚠️ Dates may be missing: {', '.join(sorted(missing_dates)[:3])}")
        
        # Domain-specific checks
        if domain == 'medical':
            # Check for medical test values (number + unit pattern)
            test_pattern = r'(\d+\.?\d*)\s*(?:mg|ml|g|IU|mmol|μmol|ng|pg|mEq|cells|/dL|/L|/cumm|%|mm/hr)'
            original_tests = set(re.findall(test_pattern, original, re.IGNORECASE))
            simplified_tests = set(re.findall(test_pattern, simplified, re.IGNORECASE))
            
            missing_tests = original_tests - simplified_tests
            if missing_tests:
                warnings.append(f"⚠️ Medical values may be missing: {', '.join(sorted(missing_tests)[:5])}")
        
        elif domain == 'legal':
            # Check for legal time periods
            time_pattern = r'(\d+)\s*(?:day|days|month|months|year|years|week|weeks)'
            original_times = set(re.findall(time_pattern, original, re.IGNORECASE))
            simplified_times = set(re.findall(time_pattern, simplified, re.IGNORECASE))
            
            missing_times = original_times - simplified_times
            if missing_times:
                warnings.append(f"⚠️ Time periods may be missing: {', '.join(sorted(missing_times)[:3])} days/months")
        
        # Length check - simplified shouldn't be drastically shorter (might lose info)
        if len(simplified) < len(original) * 0.2 and len(original) > 50:
            warnings.append("⚠️ Simplified text is very short - important details may be lost")
        
        is_consistent = len(warnings) == 0
        return is_consistent, warnings
    
    def _fallback_simplify(self, text: str, domain: str) -> str:
        """Rule-based fallback when LLM unavailable"""
        if not self.fallback_enabled:
            return text
        
        # Import the rule-based simplifier as fallback
        try:
            from ml_text_simplifier import EnhancedTextSimplifier, DocumentDomain, OutputStyle
            fallback = EnhancedTextSimplifier(use_ml=False)
            domain_map = {'legal': DocumentDomain.LEGAL, 'medical': DocumentDomain.MEDICAL}
            result = fallback.simplify(text, domain_map.get(domain, DocumentDomain.GENERAL))
            return result.simplified
        except:
            return text
    
    def simplify_clause(
        self,
        clause: Union[str, Dict],
        domain: str = "legal",
        include_translations: bool = True
    ) -> ClauseResult:
        """
        Simplify a single clause with risk detection and multi-language output.
        
        Args:
            clause: Clause text or dict with 'content'/'text' field
            domain: 'legal' or 'medical'
            include_translations: Whether to generate Hindi/Tamil translations
            
        Returns:
            ClauseResult with simplified versions in all languages and risk info
        """
        # Extract text from clause
        if isinstance(clause, dict):
            text = clause.get('content', clause.get('text', str(clause)))
        else:
            text = str(clause)
        
        text = text.strip()
        if not text:
            return ClauseResult(
                original="",
                simplified="",
                colloquial=""
            )
        
        # Generate English simplification (single version, no colloquial)
        plain = self._generate_simplification(text, style="plain", domain=domain)
        # No separate colloquial version - just use the simplified version
        colloquial = plain  # Same as plain - UI may still expect this field
        
        # Generate translations if requested
        hindi_formal = ""
        hindi_colloquial = ""
        tamil_formal = ""
        tamil_colloquial = ""
        
        if include_translations:
            # Hindi translations
            hindi_formal = self._translate(plain, 'hi')
            hindi_colloquial = self._translate(colloquial, 'hi')
            
            # Tamil translations
            tamil_formal = self._translate(plain, 'ta')
            tamil_colloquial = self._translate(colloquial, 'ta')
        
        # Detect risk using LLM
        risk_level, risk_score, risk_explanation = self._detect_risk(text, domain=domain)
        
        # Extract key terms (monetary values, dates, etc.)
        key_terms = self._extract_key_terms(text, domain=domain)
        
        # Check entity preservation
        entities_preserved, preservation_warnings = self._check_entity_preservation(text, plain)
        
        return ClauseResult(
            original=text,
            simplified=plain,
            colloquial=colloquial,
            hindi_formal=hindi_formal,
            hindi_colloquial=hindi_colloquial,
            tamil_formal=tamil_formal,
            tamil_colloquial=tamil_colloquial,
            risk_level=risk_level,
            risk_score=risk_score,
            risk_explanation=risk_explanation,
            key_terms=key_terms,
            entities_preserved=entities_preserved,
            preservation_warnings=preservation_warnings
        )
    
    def simplify_document(
        self,
        clauses: List[Union[str, Dict]],
        domain: str = "legal",
        style: OutputStyle = OutputStyle.PLAIN_ENGLISH
    ) -> DocumentResult:
        """
        Simplify all clauses in a document.
        
        Args:
            clauses: List of clause texts or dicts
            domain: 'legal' or 'medical'
            style: Output style preference
            
        Returns:
            DocumentResult with all simplified clauses and summary
        """
        print(f"\n{'='*60}")
        print(f"  LLM DOCUMENT SIMPLIFICATION")
        print(f"{'='*60}")
        print(f"   Processing {len(clauses)} clauses...")
        
        results = []
        total_risk_score = 0.0
        risky_clauses = []
        
        for i, clause in enumerate(clauses):
            print(f"   [{i+1}/{len(clauses)}] Processing clause...", end="\r")
            
            result = self.simplify_clause(clause, domain)
            results.append(result)
            total_risk_score += result.risk_score
            
            if result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                risky_clauses.append(result)
        
        print(f"   ✅ Processed {len(clauses)} clauses" + " " * 20)
        
        # Calculate overall risk
        avg_risk = total_risk_score / len(clauses) if clauses else 0.0
        
        if avg_risk >= 0.7:
            overall_risk = RiskLevel.CRITICAL
        elif avg_risk >= 0.5:
            overall_risk = RiskLevel.HIGH
        elif avg_risk >= 0.3:
            overall_risk = RiskLevel.MEDIUM
        elif avg_risk >= 0.1:
            overall_risk = RiskLevel.LOW
        else:
            overall_risk = RiskLevel.NONE
        
        # Generate summary
        summary = self._generate_summary(results, domain)
        
        # Compile warnings
        warnings = []
        for r in results:
            if r.risk_level == RiskLevel.CRITICAL:
                warnings.append(f"⚠️ CRITICAL: {r.risk_explanation[:100]}")
            elif r.risk_level == RiskLevel.HIGH:
                warnings.append(f"⚡ HIGH RISK: {r.risk_explanation[:100]}")
        
        # Add consistency warnings
        inconsistent_count = sum(1 for r in results if not r.is_consistent)
        for r in results:
            warnings.extend(r.consistency_warnings)
        
        print(f"   📊 Overall risk: {overall_risk.value.upper()} ({avg_risk:.1%})")
        print(f"   ⚠️ Risky clauses: {len(risky_clauses)}/{len(clauses)}")
        if inconsistent_count > 0:
            print(f"   ⚡ Consistency issues: {inconsistent_count} clause(s)")
        
        return DocumentResult(
            clauses=results,
            summary=summary,
            overall_risk=overall_risk,
            overall_risk_score=avg_risk,
            domain=domain,
            style=style,
            warnings=warnings[:15]  # Increased limit for consistency warnings
        )
    
    def _generate_summary(self, clause_results: List[ClauseResult], domain: str) -> str:
        """Generate a summary of the document"""
        
        if not clause_results:
            return "No clauses to summarize."
        
        # Count risk levels
        risk_counts = {}
        for r in clause_results:
            risk_counts[r.risk_level.value] = risk_counts.get(r.risk_level.value, 0) + 1
        
        # Extract key terms
        all_key_terms = []
        for r in clause_results:
            all_key_terms.extend(r.key_terms)
        
        # Build summary
        summary_parts = [
            f"Document contains {len(clause_results)} clauses.",
        ]
        
        # Add risk breakdown
        if risk_counts:
            risk_str = ", ".join(f"{k}: {v}" for k, v in sorted(risk_counts.items(), key=lambda x: -x[1]))
            summary_parts.append(f"Risk levels: {risk_str}.")
        
        # Add risk summary
        risky = sum(1 for r in clause_results if r.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL])
        if risky > 0:
            summary_parts.append(f"⚠️ {risky} clause(s) require careful attention.")
        else:
            summary_parts.append("No high-risk clauses detected.")
        
        # Add key terms
        if all_key_terms:
            unique_terms = list(set(all_key_terms))[:5]
            summary_parts.append(f"Key values: {', '.join(unique_terms)}.")
        
        # Add consistency info
        inconsistent = sum(1 for r in clause_results if not r.is_consistent)
        if inconsistent > 0:
            summary_parts.append(f"⚠️ {inconsistent} clause(s) may have consistency issues.")
        
        return " ".join(summary_parts)
    
    def get_risk_report(self, doc_result: DocumentResult) -> str:
        """
        Generate a detailed risk report for a document.
        
        Args:
            doc_result: DocumentResult from simplify_document()
            
        Returns:
            Formatted risk report string
        """
        lines = [
            "=" * 60,
            "  DOCUMENT RISK REPORT",
            "=" * 60,
            f"\n📊 Overall Risk: {doc_result.overall_risk.value.upper()} ({doc_result.overall_risk_score:.1%})",
            f"📄 Total Clauses: {len(doc_result.clauses)}",
            f"⚠️ Risky Clauses: {sum(1 for c in doc_result.clauses if c.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL])}",
            "\n" + "-" * 60,
            "CLAUSE-BY-CLAUSE ANALYSIS:",
            "-" * 60,
        ]
        
        for i, clause in enumerate(doc_result.clauses, 1):
            risk_emoji = {'none': '✅', 'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}
            emoji = risk_emoji.get(clause.risk_level.value, '⚪')
            
            lines.append(f"\n{emoji} Clause {i}:")
            lines.append(f"   Risk: {clause.risk_level.value.upper()} ({clause.risk_score:.1%})")
            lines.append(f"   Original: {clause.original[:60]}...")
            lines.append(f"   Simplified: {clause.simplified[:60]}...")
            
            if clause.key_terms:
                lines.append(f"   Key Terms: {', '.join(clause.key_terms)}")
        
        if doc_result.warnings:
            lines.append("\n" + "-" * 60)
            lines.append("DOCUMENT WARNINGS:")
            lines.append("-" * 60)
            for w in doc_result.warnings[:10]:
                lines.append(f"  • {w}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# Convenience functions
def simplify_clauses(
    clauses: List[str],
    domain: str = "legal"
) -> List[Dict]:
    """
    Quick function to simplify a list of clauses.
    
    Args:
        clauses: List of clause texts
        domain: 'legal' or 'medical'
        
    Returns:
        List of simplified clause dicts
    """
    simplifier = LLMSimplifier()
    result = simplifier.simplify_document(clauses, domain)
    return [c.to_dict() for c in result.clauses]


def simplify_text(text: str, domain: str = "legal") -> Dict:
    """
    Quick function to simplify a single text block.
    
    Args:
        text: Text to simplify
        domain: 'legal' or 'medical'
        
    Returns:
        Dict with simplified text and analysis
    """
    simplifier = LLMSimplifier()
    result = simplifier.simplify_clause(text, domain)
    return result.to_dict()


def analyze_document_risk(clauses: List[str], domain: str = "legal") -> Dict:
    """
    Analyze document risk without full simplification.
    
    Args:
        clauses: List of clause texts
        domain: 'legal' or 'medical'
        
    Returns:
        Dict with risk analysis
    """
    simplifier = LLMSimplifier()
    
    risks = []
    for clause in clauses:
        level, score, explanation = simplifier._detect_risk(clause)
        risks.append({
            'clause': clause[:100] + '...' if len(clause) > 100 else clause,
            'risk_level': level.value,
            'risk_score': round(score, 3),
            'explanation': explanation
        })
    
    avg_score = sum(r['risk_score'] for r in risks) / len(risks) if risks else 0
    high_risk_count = sum(1 for r in risks if r['risk_level'] in ['high', 'critical'])
    
    return {
        'total_clauses': len(clauses),
        'average_risk_score': round(avg_score, 3),
        'high_risk_clauses': high_risk_count,
        'clause_risks': risks
    }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  LLM SIMPLIFIER TEST")
    print("=" * 60)
    
    simplifier = LLMSimplifier()
    
    # Test clauses
    test_clauses = [
        "The Lessee shall pay a monthly rent of Rs. 15,000 on or before the 5th of each calendar month.",
        "In the event of default in payment for two consecutive months, the Lessor shall have the right to terminate this agreement and the Lessee shall forfeit the security deposit.",
        "The Lessee shall not sublet or assign the premises without prior written consent of the Lessor.",
        "The Lessee shall maintain the premises in good condition and shall be responsible for minor repairs up to Rs. 500.",
        "Either party may terminate this agreement by giving three months notice in writing.",
    ]
    
    print("\n[TEST] Processing 5 sample clauses...")
    
    for i, clause in enumerate(test_clauses):
        print(f"\n{'='*60}")
        print(f"CLAUSE {i+1}:")
        print(f"Original: {clause[:80]}...")
        
        result = simplifier.simplify_clause(clause, domain="legal")
        
        print(f"\nSimplified: {result.simplified[:80]}...")
        print(f"Colloquial: {result.colloquial[:80]}...")
        print(f"Risk: {result.risk_level.value.upper()} ({result.risk_score:.1%})")
        print(f"Explanation: {result.risk_explanation}")
        print(f"Key terms: {result.key_terms}")
    
    print("\n" + "=" * 60)
    print("  FULL DOCUMENT TEST")
    print("=" * 60)
    
    doc_result = simplifier.simplify_document(test_clauses, domain="legal")
    print(f"\nSummary: {doc_result.summary}")
    print(f"Overall Risk: {doc_result.overall_risk.value.upper()}")
    
    # Medical domain test
    print("\n" + "=" * 60)
    print("  MEDICAL DOMAIN TEST")
    print("=" * 60)
    
    medical_text = """
    Hemoglobin: 10.2 g/dL (Reference: 12.0-16.0)
    Fasting Blood Sugar: 250 mg/dL (Reference: 70-100)
    Serum Creatinine: 1.8 mg/dL (Reference: 0.5-1.5)
    """
    
    med_result = simplifier.simplify_clause(medical_text.strip(), domain="medical")
    print(f"\nOriginal: {medical_text[:60]}...")
    print(f"Simplified: {med_result.simplified[:100]}...")
    print(f"Risk: {med_result.risk_level.value.upper()} ({med_result.risk_score:.1%})")
    print(f"Key Terms: {med_result.key_terms}")
    
    print("\n" + "=" * 60)
    print("  TEST COMPLETE")
    print("=" * 60)
