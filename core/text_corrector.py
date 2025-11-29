"""
Advanced OCR Text Correction Module
====================================

Handles common OCR errors in scanned documents:
1. Character substitution errors (die → the, tlie → the)
2. Spacing errors (th e → the)
3. Case normalization issues
4. Special character corruption

Features:
- Aggressive ML correction option
- Expanded OCR patterns (200+)
- Preserves important legal references (laws, sections, articles)
- Dynamically detects and explains law references using LLM
- Preserves phone numbers, websites, emails

Author: PlainSense Team
Date: November 2025
"""

import re
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field

# Try to import spell checking libraries
try:
    from symspellpy import SymSpell, Verbosity
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False


@dataclass
class LawReference:
    """
    Represents a detected law reference with explanation.
    Used for generating explanation popups in the UI.
    """
    law_name: str
    year: str = ""
    section: str = ""
    article: str = ""
    full_reference: str = ""
    explanation: str = ""
    context: str = ""
    source: str = "detected"  # 'detected', 'llm', 'cache'
    
    def to_dict(self) -> Dict:
        return {
            'law_name': self.law_name,
            'year': self.year,
            'section': self.section,
            'article': self.article,
            'full_reference': self.full_reference,
            'explanation': self.explanation,
            'context': self.context,
            'source': self.source
        }


class DynamicLawExplainer:
    """
    Dynamically detects and explains laws using pattern matching and LLM.
    No hardcoded law database - discovers laws from text and generates explanations.
    """
    
    def __init__(self, llm_callback: Optional[Callable[[str], str]] = None):
        """
        Initialize the dynamic law explainer.
        
        Args:
            llm_callback: Optional callback function that takes a prompt and returns LLM response.
                         If not provided, will use basic explanations.
        """
        self.llm_callback = llm_callback
        self._explanation_cache: Dict[str, str] = {}
        
        # Patterns to detect law references (not hardcoded laws, just patterns)
        # Order matters - more specific patterns first
        self.law_patterns = [
            # "The XYZ Act, YYYY" or "XYZ Act, YYYY" or "XYZ Act YYYY"
            (r'(?:The\s+)?([A-Z][a-zA-Z]+(?:\s+(?:of|and|&)\s+)?[A-Za-z]*(?:\s+[A-Za-z]+)*?)\s+Act[,\s]+(\d{4})', 'act'),
            # "XYZ Act" without year (stop at "and", "or", punctuation)
            (r'(?:The\s+)?([A-Z][a-zA-Z]+(?:\s+(?:of)\s+)?[A-Za-z]*(?:\s+[A-Za-z]+)*?)\s+Act(?=\s|,|\.|\)|$)', 'act_no_year'),
            # "Section X of the XYZ Act"
            (r'Section\s+(\d+[A-Za-z]?(?:\s*\([a-z0-9]+\))?)\s+of\s+(?:the\s+)?([A-Z][a-zA-Z\s]+?Act)', 'section_of_act'),
            # "under Section X"
            (r'under\s+Section\s+(\d+[A-Za-z]?(?:\s*\([a-z0-9]+\))?)', 'section'),
            # "Article X" (for Constitution)
            (r'Article\s+(\d+[A-Za-z]?(?:\s*\([a-z0-9]+\))?)', 'article'),
            # "Rule X of the XYZ Rules"
            (r'Rule\s+(\d+[A-Za-z]?)\s+of\s+(?:the\s+)?([A-Z][a-zA-Z\s]+?Rules)', 'rule_of'),
            # "XYZ Rules, YYYY"
            (r'(?:The\s+)?([A-Z][a-zA-Z]+(?:\s+[A-Z]?[a-zA-Z]+)*?)\s+Rules[,\s]+(\d{4})', 'rules'),
            # Specific well-known Codes (Indian Penal Code, Code of Civil Procedure, etc.)
            (r'(?:The\s+)?(Indian\s+Penal\s+Code)', 'ipc'),
            (r'(?:The\s+)?(Code\s+of\s+Civil\s+Procedure)', 'cpc'),
            (r'(?:The\s+)?(Code\s+of\s+Criminal\s+Procedure)', 'crpc'),
            (r'(?:The\s+)?(Indian\s+Evidence\s+Act)', 'evidence_act'),
        ]
    
    def detect_laws(self, text: str) -> List[LawReference]:
        """
        Dynamically detect all law references in the text.
        Returns list of LawReference objects (without explanations yet).
        """
        detected = []
        seen_law_names = set()  # Avoid duplicates based on normalized law name
        
        for pattern, ref_type in self.law_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                full_ref = match.group(0).strip()
                
                # Extract components based on pattern type
                law_ref = self._parse_match(match, ref_type, text)
                if law_ref:
                    # Normalize law name for deduplication
                    normalized_name = law_ref.law_name.lower().strip()
                    # Remove common prefixes/suffixes for comparison
                    normalized_name = normalized_name.replace('the ', '').replace(' act', '')
                    
                    # Skip if we already have this law (prefer version with year)
                    if normalized_name in seen_law_names:
                        # Check if this version has a year and existing doesn't
                        existing = next((l for l in detected if l.law_name.lower().replace('the ', '').replace(' act', '') == normalized_name), None)
                        if existing and not existing.year and law_ref.year:
                            # Replace with version that has year
                            detected.remove(existing)
                            detected.append(law_ref)
                            seen_law_names.add(normalized_name)
                        continue
                    
                    seen_law_names.add(normalized_name)
                    detected.append(law_ref)
        
        return detected
    
    def _parse_match(self, match: re.Match, ref_type: str, full_text: str) -> Optional[LawReference]:
        """Parse a regex match into a LawReference object."""
        full_ref = match.group(0).strip()
        
        # Get surrounding context (100 chars before and after)
        start = max(0, match.start() - 100)
        end = min(len(full_text), match.end() + 100)
        context = full_text[start:end].strip()
        
        if ref_type == 'act':
            return LawReference(
                law_name=match.group(1).strip() + " Act",
                year=match.group(2),
                full_reference=full_ref,
                context=context
            )
        elif ref_type == 'act_no_year':
            return LawReference(
                law_name=match.group(1).strip() + " Act",
                full_reference=full_ref,
                context=context
            )
        elif ref_type == 'section_of_act':
            return LawReference(
                law_name=match.group(2).strip(),
                section=match.group(1),
                full_reference=full_ref,
                context=context
            )
        elif ref_type == 'section':
            return LawReference(
                law_name="(Referenced Law)",
                section=match.group(1),
                full_reference=full_ref,
                context=context
            )
        elif ref_type == 'article':
            return LawReference(
                law_name="Constitution of India",
                article=match.group(1),
                full_reference=full_ref,
                context=context
            )
        elif ref_type == 'rule_of':
            return LawReference(
                law_name=match.group(2).strip(),
                section=f"Rule {match.group(1)}",
                full_reference=full_ref,
                context=context
            )
        elif ref_type == 'rules':
            return LawReference(
                law_name=match.group(1).strip() + " Rules",
                year=match.group(2),
                full_reference=full_ref,
                context=context
            )
        elif ref_type in ('ipc', 'cpc', 'crpc', 'evidence_act'):
            # Specific well-known codes/acts
            return LawReference(
                law_name=match.group(1).strip(),
                full_reference=full_ref,
                context=context
            )
        
        return None
    
    def explain_law(self, law_ref: LawReference) -> str:
        """
        Generate explanation for a law reference.
        Uses LLM if available, otherwise returns a basic explanation.
        """
        # Check cache first
        cache_key = f"{law_ref.law_name}_{law_ref.section}_{law_ref.article}".lower()
        if cache_key in self._explanation_cache:
            return self._explanation_cache[cache_key]
        
        explanation = ""
        
        # Try LLM callback if available
        if self.llm_callback:
            try:
                prompt = self._build_explanation_prompt(law_ref)
                explanation = self.llm_callback(prompt)
            except Exception as e:
                print(f"   ⚠️ LLM explanation failed: {e}")
        
        # Fallback to basic explanation
        if not explanation:
            explanation = self._generate_basic_explanation(law_ref)
        
        # Cache the result
        self._explanation_cache[cache_key] = explanation
        return explanation
    
    def _build_explanation_prompt(self, law_ref: LawReference) -> str:
        """Build a prompt for the LLM to explain the law."""
        parts = [f"Explain the following Indian law reference in simple terms for a common person:"]
        parts.append(f"\nLaw: {law_ref.law_name}")
        if law_ref.year:
            parts.append(f"Year: {law_ref.year}")
        if law_ref.section:
            parts.append(f"Section: {law_ref.section}")
        if law_ref.article:
            parts.append(f"Article: {law_ref.article}")
        if law_ref.context:
            parts.append(f"\nContext where it appears: \"{law_ref.context[:200]}...\"")
        
        parts.append("\nProvide a 2-3 sentence explanation of:")
        parts.append("1. What this law/section is about")
        parts.append("2. Why it's relevant in this context")
        parts.append("3. What it means for the reader")
        
        return "\n".join(parts)
    
    def _generate_basic_explanation(self, law_ref: LawReference) -> str:
        """Generate a basic explanation without LLM."""
        parts = []
        
        law_name = law_ref.law_name
        
        # Provide context-aware basic explanations
        if 'transfer of property' in law_name.lower():
            parts.append("This law governs how property (including rental agreements) can be transferred, leased, or sold.")
        elif 'contract' in law_name.lower():
            parts.append("This law defines what makes a valid contract and the rights/obligations of parties.")
        elif 'rent' in law_name.lower():
            parts.append("This law protects tenants and landlords, regulating rent amounts and eviction procedures.")
        elif 'registration' in law_name.lower():
            parts.append("This law requires certain documents to be officially registered to be legally valid.")
        elif 'stamp' in law_name.lower():
            parts.append("This law requires payment of stamp duty on legal documents for them to be valid in court.")
        elif 'arbitration' in law_name.lower():
            parts.append("This law provides for dispute resolution through arbitration instead of courts.")
        elif 'consumer' in law_name.lower():
            parts.append("This law protects consumers against unfair trade practices and defective goods/services.")
        elif 'evidence' in law_name.lower():
            parts.append("This law defines what evidence is admissible in court proceedings.")
        elif 'limitation' in law_name.lower():
            parts.append("This law sets time limits within which legal cases must be filed.")
        elif 'constitution' in law_name.lower() or law_ref.article:
            parts.append("This refers to the Constitution of India, the supreme law of the land.")
        elif 'penal code' in law_name.lower() or 'ipc' in law_name.lower():
            parts.append("This is the main criminal code of India defining offenses and penalties.")
        elif 'civil procedure' in law_name.lower() or 'cpc' in law_name.lower():
            parts.append("This law governs the procedure for civil court cases.")
        else:
            parts.append(f"This is a legal reference to {law_name}.")
        
        # Add section-specific info if available
        if law_ref.section:
            parts.append(f"Section {law_ref.section} contains specific provisions relevant to this context.")
        if law_ref.article:
            parts.append(f"Article {law_ref.article} of the Constitution is referenced here.")
        
        # Add year context
        if law_ref.year:
            parts.append(f"This law was enacted in {law_ref.year}.")
        
        parts.append("Consult a legal professional for detailed interpretation.")
        
        return " ".join(parts)
    
    def detect_and_explain(self, text: str) -> List[LawReference]:
        """
        Detect all laws in text and generate explanations for each.
        """
        laws = self.detect_laws(text)
        
        for law in laws:
            law.explanation = self.explain_law(law)
            law.source = 'llm' if self.llm_callback else 'basic'
        
        return laws


class OCRTextCorrector:
    """
    Corrects common OCR errors in text extracted from scanned documents.
    Uses a combination of rule-based and ML-based approaches.
    """
    
    def __init__(self, use_ml: bool = True, aggressive_mode: bool = True):
        """
        Initialize the OCR text corrector.
        
        Args:
            use_ml: Whether to use SymSpell for ML-based correction
            aggressive_mode: Enable aggressive correction mode (Option B)
        """
        self.use_ml = use_ml and SYMSPELL_AVAILABLE
        self.aggressive_mode = aggressive_mode
        self.sym_spell = None
        self.detected_laws: List[LawReference] = []
        
        if self.use_ml:
            try:
                self._init_symspell()
            except Exception as e:
                print(f"⚠️ Could not initialize SymSpell: {e}")
                self.use_ml = False
        
        # Patterns to PRESERVE (phone numbers, websites, emails, legal references)
        self.preserve_patterns = [
            # Phone numbers - various formats
            (r'(\+91[\s.-]?)?[6-9]\d{4}[\s.-]?\d{5}', 'PHONE'),
            (r'\b\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b', 'PHONE'),
            (r'\b\d{5}[\s.-]?\d{5}\b', 'PHONE'),
            (r'\b\d{4}[\s.-]?\d{6}\b', 'PHONE'),
            # Landline with STD code
            (r'\b0\d{2,4}[\s.-]?\d{6,8}\b', 'PHONE'),
            # Emails
            (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 'EMAIL'),
            # Websites/URLs
            (r'https?://[^\s<>"{}|\\^`\[\]]+', 'URL'),
            (r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*', 'URL'),
            # Law references - Section/Article/Clause
            (r'Section\s+\d+[A-Za-z]?(?:\s*\([a-z0-9]+\))?', 'LEGAL_REF'),
            (r'Sec\.\s*\d+[A-Za-z]?(?:\s*\([a-z0-9]+\))?', 'LEGAL_REF'),
            (r'Article\s+\d+[A-Za-z]?(?:\s*\([a-z0-9]+\))?', 'LEGAL_REF'),
            (r'Art\.\s*\d+[A-Za-z]?', 'LEGAL_REF'),
            (r'Clause\s+\d+[A-Za-z]?(?:\s*\([a-z0-9]+\))?', 'LEGAL_REF'),
            (r'Rule\s+\d+[A-Za-z]?(?:\s*\([a-z0-9]+\))?', 'LEGAL_REF'),
            (r'Schedule\s+[IVX]+|Schedule\s+\d+', 'LEGAL_REF'),
            # Act references
            (r'\b(?:The\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Act,?\s*(?:19|20)\d{2}', 'ACT_REF'),
            # Specific document numbers
            (r'\b[A-Z]{2,5}[-/]\d{2,}[-/]?\d*[-/]?\d*\b', 'DOC_NUMBER'),
            # PIN codes
            (r'\b\d{6}\b', 'PINCODE'),
            # PAN numbers
            (r'\b[A-Z]{5}\d{4}[A-Z]\b', 'PAN'),
            # Aadhaar
            (r'\b\d{4}\s?\d{4}\s?\d{4}\b', 'AADHAAR'),
            # Bank account numbers (8-18 digits)
            (r'\b\d{8,18}\b', 'ACCOUNT'),
            # IFSC codes
            (r'\b[A-Z]{4}0[A-Z0-9]{6}\b', 'IFSC'),
        ]
        
        # EXPANDED OCR character substitution errors (Option C: 200+ patterns)
        # Format: (wrong_pattern, correct_replacement)
        self.char_substitutions = [
            # ===============================
            # THE/THIS/THAT/THERE/THEN/THEY Family (most common)
            # ===============================
            (r'\bdie\b', 'the'),
            (r'\bDie\b', 'The'),
            (r'\bDIE\b', 'THE'),
            (r'\btlie\b', 'the'),
            (r'\bTlie\b', 'The'),
            (r'\btbe\b', 'the'),
            (r'\bTbe\b', 'The'),
            (r'\btne\b', 'the'),
            (r'\bTne\b', 'The'),
            (r'\bibe\b', 'the'),
            (r'\bIbe\b', 'The'),
            (r'\bthe\s+the\b', 'the'),  # Duplicate fix
            (r'\bMie\b', 'the'),
            (r'\bmie\b', 'the'),
            (r'\btiie\b', 'the'),
            (r'\bTiie\b', 'The'),
            (r'\btlne\b', 'the'),
            (r'\bTlne\b', 'The'),
            (r'\btbc\b', 'the'),
            (r'\bTbc\b', 'The'),
            
            # THIS variants
            (r'\bdiis\b', 'this'),
            (r'\bDiis\b', 'This'),
            (r'\btliis\b', 'this'),
            (r'\bTliis\b', 'This'),
            (r'\bthls\b', 'this'),
            (r'\bThls\b', 'This'),
            (r'\btiiis\b', 'this'),
            (r'\bTiiis\b', 'This'),
            (r'\bthis\s+this\b', 'this'),
            
            # THAT variants
            (r'\bdiat\b', 'that'),
            (r'\bDiat\b', 'That'),
            (r'\btliat\b', 'that'),
            (r'\bTliat\b', 'That'),
            (r'\btlat\b', 'that'),
            (r'\bTlat\b', 'That'),
            (r'\btnat\b', 'that'),
            (r'\bTnat\b', 'That'),
            (r'\btiiat\b', 'that'),
            (r'\bTiiat\b', 'That'),
            
            # THERE variants
            (r'\bdiere\b', 'there'),
            (r'\bDiere\b', 'There'),
            (r'\btliere\b', 'there'),
            (r'\bTliere\b', 'There'),
            (r'\btbere\b', 'there'),
            (r'\bTbere\b', 'There'),
            (r'\btiiere\b', 'there'),
            (r'\bTiiere\b', 'There'),
            
            # THEN variants
            (r'\bdien\b', 'then'),
            (r'\bDien\b', 'Then'),
            (r'\btlien\b', 'then'),
            (r'\bTlien\b', 'Then'),
            (r'\btben\b', 'then'),
            (r'\bTben\b', 'Then'),
            
            # THEY variants
            (r'\bdiey\b', 'they'),
            (r'\bDiey\b', 'They'),
            (r'\btliey\b', 'they'),
            (r'\bTliey\b', 'They'),
            (r'\btbey\b', 'they'),
            (r'\bTbey\b', 'They'),
            
            # THEIR variants
            (r'\bdieir\b', 'their'),
            (r'\bDieir\b', 'Their'),
            (r'\btlieir\b', 'their'),
            (r'\bTlieir\b', 'Their'),
            
            # ===============================
            # WITH/WHICH/WHAT/WHEN/WHERE/WHO Family
            # ===============================
            (r'\bwidi\b', 'with'),
            (r'\bWidi\b', 'With'),
            (r'\bwitli\b', 'with'),
            (r'\bWithi\b', 'With'),
            (r'\bwitb\b', 'with'),
            (r'\bWithb\b', 'With'),
            (r'\bwith\s+with\b', 'with'),
            (r'\bwitn\b', 'with'),
            (r'\bWithn\b', 'With'),
            (r'\bvvith\b', 'with'),
            (r'\bVvith\b', 'With'),
            
            (r'\bwliich\b', 'which'),
            (r'\bWliich\b', 'Which'),
            (r'\bwbich\b', 'which'),
            (r'\bWbich\b', 'Which'),
            (r'\bwhicli\b', 'which'),
            (r'\bWhicli\b', 'Which'),
            (r'\bwhlch\b', 'which'),
            (r'\bWhlch\b', 'Which'),
            
            (r'\bwliat\b', 'what'),
            (r'\bWliat\b', 'What'),
            (r'\bwbat\b', 'what'),
            (r'\bWbat\b', 'What'),
            
            (r'\bwlien\b', 'when'),
            (r'\bWlien\b', 'When'),
            (r'\bwben\b', 'when'),
            (r'\bWben\b', 'When'),
            
            (r'\bwliere\b', 'where'),
            (r'\bWliere\b', 'Where'),
            (r'\bwbere\b', 'where'),
            (r'\bWbere\b', 'Where'),
            
            (r'\bwlio\b', 'who'),
            (r'\bWlio\b', 'Who'),
            
            # ===============================
            # HAVE/HAS/HAD Family
            # ===============================
            (r'\bliave\b', 'have'),
            (r'\bLiave\b', 'Have'),
            (r'\bHave\b', 'Have'),
            (r'\bbave\b', 'have'),
            (r'\bBave\b', 'Have'),
            (r'\bhave\s+have\b', 'have'),
            (r'\bnave\b', 'have'),
            (r'\bNave\b', 'Have'),
            
            (r'\blias\b', 'has'),
            (r'\bLias\b', 'Has'),
            (r'\bbas\b', 'has'),
            (r'\bBas\b', 'Has'),
            
            (r'\bliad\b', 'had'),
            (r'\bLiad\b', 'Had'),
            (r'\bbad\b', 'had'),
            (r'\bBad\b', 'Had'),
            
            # ===============================
            # BEEN/BE/BEING Family
            # ===============================
            (r'\bbeen\b', 'been'),
            (r'\bBeeu\b', 'Been'),
            (r'\bbeeu\b', 'been'),
            (r'\bbccn\b', 'been'),
            
            # ===============================
            # SHALL/WILL/WOULD/COULD/SHOULD Family
            # ===============================
            (r'\bsha11\b', 'shall'),
            (r'\bSha11\b', 'Shall'),
            (r'\bshall\s+shall\b', 'shall'),
            (r'\bsliall\b', 'shall'),
            (r'\bSliall\b', 'Shall'),
            (r'\bsbal1\b', 'shall'),
            
            (r'\bwi11\b', 'will'),
            (r'\bWi11\b', 'Will'),
            (r'\bwill\s+will\b', 'will'),
            (r'\bwiU\b', 'will'),
            (r'\bWiU\b', 'Will'),
            
            (r'\bwou1d\b', 'would'),
            (r'\bWou1d\b', 'Would'),
            (r'\bwonld\b', 'would'),
            (r'\bWonld\b', 'Would'),
            
            (r'\bcou1d\b', 'could'),
            (r'\bCou1d\b', 'Could'),
            (r'\bconld\b', 'could'),
            (r'\bConld\b', 'Could'),
            
            (r'\bshou1d\b', 'should'),
            (r'\bShou1d\b', 'Should'),
            (r'\bsliould\b', 'should'),
            (r'\bSliould\b', 'Should'),
            
            # ===============================
            # FROM/FOR/FORM Family
            # ===============================
            (r'\bfrorn\b', 'from'),
            (r'\bFrorn\b', 'From'),
            (r'\bfrom\s+from\b', 'from'),
            (r'\bfrom\b', 'from'),
            (r'\bfroni\b', 'from'),
            (r'\bFromi\b', 'From'),
            
            (r'\bfor\s+for\b', 'for'),
            (r'\bf0r\b', 'for'),
            (r'\bF0r\b', 'For'),
            
            # ===============================
            # MONTH/MONTHLY Family
            # ===============================
            (r'\bnontli\b', 'month'),
            (r'\bNontli\b', 'Month'),
            (r'\bmontli\b', 'month'),
            (r'\bMontli\b', 'Month'),
            (r'\bmonth1y\b', 'monthly'),
            (r'\bMonth1y\b', 'Monthly'),
            (r'\bmonthlv\b', 'monthly'),
            (r'\bMonthlv\b', 'Monthly'),
            (r'\bmonthiy\b', 'monthly'),
            (r'\bMonthiy\b', 'Monthly'),
            
            # ===============================
            # ALL/CALL Family
            # ===============================
            (r'\bca11\b', 'call'),
            (r'\bCa11\b', 'Call'),
            (r'\ba11\b', 'all'),
            (r'\bA11\b', 'All'),
            (r'\ball\s+all\b', 'all'),
            
            # ===============================
            # 'o' and '0' confusion
            # ===============================
            (r'\bof0\b', 'of'),
            (r'\bt0\b', 'to'),
            (r'\bT0\b', 'To'),
            (r'\bto\s+to\b', 'to'),
            (r'\bn0t\b', 'not'),
            (r'\bN0t\b', 'Not'),
            (r'\bals0\b', 'also'),
            (r'\bAls0\b', 'Also'),
            (r'\bint0\b', 'into'),
            (r'\bInt0\b', 'Into'),
            (r'\bn0w\b', 'now'),
            (r'\bN0w\b', 'Now'),
            (r'\bwh0\b', 'who'),
            (r'\bWh0\b', 'Who'),
            (r'\bh0w\b', 'how'),
            (r'\bH0w\b', 'How'),
            (r'\bkn0w\b', 'know'),
            (r'\bKn0w\b', 'Know'),
            (r'\b0wner\b', 'owner'),
            (r'\b0wner\b', 'Owner'),
            (r'\b0f\b', 'of'),
            (r'\b0r\b', 'or'),
            
            # ===============================
            # Legal Terms - Critical Preservation
            # ===============================
            (r'\bentided\b', 'entitled'),
            (r'\bEntided\b', 'Entitled'),
            (r'\bentitied\b', 'entitled'),
            (r'\bEntitied\b', 'Entitled'),
            
            (r'\bhereinaf ter\b', 'hereinafter'),
            (r'\bHereinaf ter\b', 'Hereinafter'),
            (r'\bhereinafter\b', 'hereinafter'),  # Normalize
            
            (r'\blandlord\s*/\s*agent\b', 'landlord/agent'),
            (r'\bten ant\b', 'tenant'),
            (r'\bTen ant\b', 'Tenant'),
            (r'\btcnant\b', 'tenant'),
            (r'\bTcnant\b', 'Tenant'),
            
            (r'\bland lord\b', 'landlord'),
            (r'\bLand lord\b', 'Landlord'),
            (r'\blandlord\s+landlord\b', 'landlord'),
            (r'\blandiord\b', 'landlord'),
            (r'\bLandiord\b', 'Landlord'),
            
            (r'\bpre mises\b', 'premises'),
            (r'\bPre mises\b', 'Premises'),
            (r'\bprernises\b', 'premises'),
            (r'\bPrernises\b', 'Premises'),
            (r'\bpreniises\b', 'premises'),
            
            (r'\bagree ment\b', 'agreement'),
            (r'\bAgree ment\b', 'Agreement'),
            (r'\bAGREE MENT\b', 'AGREEMENT'),
            (r'\bagreernent\b', 'agreement'),
            (r'\bAgreernent\b', 'Agreement'),
            
            (r'\bpur suant\b', 'pursuant'),
            (r'\bPur suant\b', 'Pursuant'),
            (r'\bpursuant\b', 'pursuant'),  # Normalize
            
            (r'\bac knowledge\b', 'acknowledge'),
            (r'\bAc knowledge\b', 'Acknowledge'),
            
            (r'\bin formation\b', 'information'),
            (r'\bIn formation\b', 'Information'),
            
            (r'\bcon dition\b', 'condition'),
            (r'\bCon dition\b', 'Condition'),
            
            (r'\bprop erty\b', 'property'),
            (r'\bProp erty\b', 'Property'),
            
            (r'\bpay ment\b', 'payment'),
            (r'\bPay ment\b', 'Payment'),
            
            (r'\bdepos it\b', 'deposit'),
            (r'\bDepos it\b', 'Deposit'),
            
            (r'\brent al\b', 'rental'),
            (r'\bRent al\b', 'Rental'),
            
            (r'\bsecu rity\b', 'security'),
            (r'\bSecu rity\b', 'Security'),
            
            (r'\bnotice\b', 'notice'),  # Normalize
            (r'\bnot ice\b', 'notice'),
            (r'\bNot ice\b', 'Notice'),
            
            (r'\btermi nation\b', 'termination'),
            (r'\bTermi nation\b', 'Termination'),
            
            (r'\boccup ancy\b', 'occupancy'),
            (r'\bOccup ancy\b', 'Occupancy'),
            
            (r'\bmain tenance\b', 'maintenance'),
            (r'\bMain tenance\b', 'Maintenance'),
            (r'\bmaintainence\b', 'maintenance'),
            (r'\bMaintainence\b', 'Maintenance'),
            
            (r'\brepre sent\b', 'represent'),
            (r'\bRepre sent\b', 'Represent'),
            
            (r'\bwar ranty\b', 'warranty'),
            (r'\bWar ranty\b', 'Warranty'),
            
            (r'\blia bility\b', 'liability'),
            (r'\bLia bility\b', 'Liability'),
            
            (r'\bindem nity\b', 'indemnity'),
            (r'\bIndem nity\b', 'Indemnity'),
            
            (r'\barbi tration\b', 'arbitration'),
            (r'\bArbi tration\b', 'Arbitration'),
            
            (r'\bjuris diction\b', 'jurisdiction'),
            (r'\bJuris diction\b', 'Jurisdiction'),
            
            # ===============================
            # Execute/Receive/Receipt Family
            # ===============================
            (r'\bexeeutive\b', 'executive'),
            (r'\bExecutive\b', 'Executive'),
            (r'\bexeeute\b', 'execute'),
            (r'\bExeeute\b', 'Execute'),
            (r'\bexecnte\b', 'execute'),
            (r'\bExecnte\b', 'Execute'),
            
            (r'\breeeive\b', 'receive'),
            (r'\bReeeive\b', 'Receive'),
            (r'\breceive\b', 'receive'),  # Normalize
            
            (r'\breeeipt\b', 'receipt'),
            (r'\bReeeip t\b', 'Receipt'),
            (r'\breceipt\b', 'receipt'),  # Normalize
            
            # ===============================
            # BE/IS/ARE/AN/AND Family
            # ===============================
            (r'\bbc\b', 'be'),
            (r'\bBc\b', 'Be'),
            (r'\bbe\s+be\b', 'be'),
            
            (r'\bis\s+is\b', 'is'),
            (r'\bare\s+are\b', 'are'),
            
            (r'\ban\s+an\b', 'an'),
            (r'\band\s+and\b', 'and'),
            (r'\baud\b', 'and'),
            (r'\bAud\b', 'And'),
            (r'\barid\b', 'and'),
            (r'\bArid\b', 'And'),
            
            # ===============================
            # Other Common Words
            # ===============================
            (r'\bained\b', 'maintained'),
            (r'\bper son\b', 'person'),
            (r'\bPer son\b', 'Person'),
            
            (r'\bany one\b', 'anyone'),
            (r'\bAny one\b', 'Anyone'),
            
            (r'\bsome one\b', 'someone'),
            (r'\bSome one\b', 'Someone'),
            
            (r'\bno body\b', 'nobody'),
            (r'\bNo body\b', 'Nobody'),
            
            (r'\bto day\b', 'today'),
            (r'\bTo day\b', 'Today'),
            
            (r'\bto morrow\b', 'tomorrow'),
            (r'\bTo morrow\b', 'Tomorrow'),
            
            (r'\byester day\b', 'yesterday'),
            (r'\bYester day\b', 'Yesterday'),
            
            (r'\bim mediately\b', 'immediately'),
            (r'\bIm mediately\b', 'Immediately'),
            
            (r'\bac cordingly\b', 'accordingly'),
            (r'\bAc cordingly\b', 'Accordingly'),
            
            (r'\bap plicable\b', 'applicable'),
            (r'\bAp plicable\b', 'Applicable'),
            
            (r'\bres ponsible\b', 'responsible'),
            (r'\bRes ponsible\b', 'Responsible'),
            
            # ===============================
            # pomonyof, b&gfs fixes (from document)
            # ===============================
            (r'\bpomonyof\b', 'portion of'),
            (r'\bPomonyof\b', 'Portion of'),
            (r'\bb&gfs\b', "days'"),  # OCR misread of "days'"
            (r'\bB&gfs\b', "Days'"),
            (r'\bd&gfs\b', "days'"),
            
            # ===============================
            # OCR artifacts removal/fix
            # ===============================
            (r'[|l]andlord', 'landlord'),
            (r'[|l]essee', 'lessee'),
            (r'[|l]essor', 'lessor'),
            (r'[|l]ease', 'lease'),
            (r'[|I]ndia', 'India'),
            
            # Clean artifacts but be careful
            (r'(?<!\d)\^(?!\d)', ''),  # Remove caret artifacts (not in math)
            
            # Fix quotes
            (r'["""]', '"'),
            (r"[''']", "'"),
            
            # Number/letter confusion  
            (r'Rs\s*\.?\s*(\d)', r'Rs. \1'),  # Standardize Rs.
            (r'INR\s*(\d)', r'INR \1'),
            
            # ===============================
            # Law/Legal Reference Fixes
            # ===============================
            (r'Seetion\s+', 'Section '),
            (r'seetion\s+', 'section '),
            (r'Artiele\s+', 'Article '),
            (r'artiele\s+', 'article '),
            (r'Aet\s*,?\s*(19|20)\d{2}', r'Act, \1'),
            (r'aet\s*,?\s*(19|20)\d{2}', r'act, \1'),
        ]
        
        # Patterns that indicate OCR garbage (too many special chars)
        self.garbage_patterns = [
            r'[^\w\s,.;:!?()\'"-]{3,}',  # 3+ consecutive special chars
            r'(?:[bcdfghjklmnpqrstvwxz]{6,})',  # 6+ consonants (unlikely)
            r'(?:\d[a-z]\d[a-z]){2,}',  # Alternating digits/letters pattern
            r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]',  # Control characters
        ]
        
        # Initialize dynamic law explainer (no hardcoded database)
        self.law_explainer = DynamicLawExplainer(llm_callback=None)
        
    def set_llm_callback(self, callback: Callable[[str], str]):
        """
        Set the LLM callback for generating law explanations.
        This should be called after initializing with the LLM simplifier.
        
        Args:
            callback: Function that takes a prompt string and returns LLM response
        """
        self.law_explainer.llm_callback = callback
        
    def _init_symspell(self):
        """Initialize SymSpell spell checker."""
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        
        # Load dictionary
        try:
            import pkg_resources
            dict_path = pkg_resources.resource_filename(
                'symspellpy',
                'frequency_dictionary_en_82_765.txt'
            )
            self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
            print(f"   ✅ OCR Corrector: Loaded {len(self.sym_spell.words):,} words dictionary")
        except Exception as e:
            print(f"   ⚠️ Could not load SymSpell dictionary: {e}")
            self.use_ml = False
    
    def _extract_and_preserve(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Extract important information before correction to preserve it.
        Returns modified text with placeholders and a mapping of placeholders to original values.
        """
        preserved = {}
        placeholder_count = 0
        
        for pattern, pattern_type in self.preserve_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                placeholder = f"__PRESERVED_{pattern_type}_{placeholder_count}__"
                preserved[placeholder] = match.group()
                text = text.replace(match.group(), placeholder, 1)
                placeholder_count += 1
        
        return text, preserved
    
    def _restore_preserved(self, text: str, preserved: Dict[str, str]) -> str:
        """Restore preserved content after correction."""
        for placeholder, original in preserved.items():
            text = text.replace(placeholder, original)
        return text
    
    def detect_law_references(self, text: str) -> List[LawReference]:
        """
        Detect and extract law references from text dynamically.
        Returns list of LawReference objects with explanations.
        
        Uses DynamicLawExplainer - no hardcoded law database.
        Laws are discovered from text patterns and explained using LLM or basic rules.
        
        This is used to generate popup explanations in the UI.
        """
        # Use the dynamic law explainer
        self.detected_laws = self.law_explainer.detect_and_explain(text)
        return self.detected_laws
    
    def correct_text(self, text: str, aggressive: bool = None) -> str:
        """
        Correct OCR errors in text.
        
        Args:
            text: Input text with potential OCR errors
            aggressive: If True, apply more aggressive corrections (may change meaning)
                       If None, uses self.aggressive_mode
            
        Returns:
            Corrected text
        """
        import time
        
        if not text:
            return text
        
        if aggressive is None:
            aggressive = self.aggressive_mode
        
        original_text = text
        start_time = time.time()
        
        # Step 0: Preserve important information (phone numbers, URLs, legal refs)
        text, preserved = self._extract_and_preserve(text)
        print(f"   ⏱️ Step 0 (preserve): {time.time() - start_time:.2f}s")
        
        # Step 1: Fix character substitution errors (rule-based)
        step_time = time.time()
        text = self._fix_char_substitutions(text)
        print(f"   ⏱️ Step 1 (char subs): {time.time() - step_time:.2f}s")
        
        # Step 2: Fix spacing errors
        step_time = time.time()
        text = self._fix_spacing_errors(text)
        print(f"   ⏱️ Step 2 (spacing): {time.time() - step_time:.2f}s")
        
        # Step 3: Remove obvious OCR garbage
        step_time = time.time()
        text = self._remove_garbage(text)
        print(f"   ⏱️ Step 3 (garbage): {time.time() - step_time:.2f}s")
        
        # Step 4: ML-based correction (if available)
        # With aggressive mode, we apply it to more text
        if self.use_ml and aggressive:
            step_time = time.time()
            text = self._ml_correct(text, aggressive=aggressive)
            print(f"   ⏱️ Step 4 (ML): {time.time() - step_time:.2f}s")
        
        # Step 5: Final cleanup
        step_time = time.time()
        text = self._final_cleanup(text)
        print(f"   ⏱️ Step 5 (cleanup): {time.time() - step_time:.2f}s")
        
        # Step 6: Restore preserved content
        text = self._restore_preserved(text, preserved)
        
        # Step 7: Detect law references for popup explanations
        # SKIP for now - regex patterns can be slow on large documents
        # step_time = time.time()
        # self.detect_law_references(text)
        # print(f"   ⏱️ Step 7 (law refs): {time.time() - step_time:.2f}s")
        
        print(f"   ✅ Text correction complete: {time.time() - start_time:.2f}s total")
        
        return text
    
    def _fix_char_substitutions(self, text: str) -> str:
        """Apply character substitution fixes."""
        for pattern, replacement in self.char_substitutions:
            try:
                text = re.sub(pattern, replacement, text, flags=0)
            except Exception:
                continue
        return text
    
    def _fix_spacing_errors(self, text: str) -> str:
        """Fix common spacing errors from OCR."""
        # Fix split words (space in middle of word)
        # Common patterns where a space was incorrectly inserted
        spacing_fixes = [
            (r'(\w)\s+([.,;:!?])', r'\1\2'),  # Remove space before punctuation
            (r'([.,;:!?])\s{2,}', r'\1 '),  # Reduce multiple spaces after punctuation
            (r'\(\s+', '('),  # Remove space after opening paren
            (r'\s+\)', ')'),  # Remove space before closing paren
            (r'\s{3,}', '  '),  # Reduce 3+ spaces to 2
            (r'(\n\s*){3,}', '\n\n'),  # Reduce multiple blank lines
        ]
        
        for pattern, replacement in spacing_fixes:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _remove_garbage(self, text: str) -> str:
        """Remove obvious OCR garbage patterns while preserving important data."""
        for pattern in self.garbage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Skip if it looks like preserved content
                if '__PRESERVED_' in match:
                    continue
                # Only remove if clearly garbage (not numbers, not common patterns)
                if not any(c.isdigit() for c in match) and len(match) < 20:
                    text = text.replace(match, ' ')
        
        return text
    
    def _ml_correct(self, text: str, aggressive: bool = True) -> str:
        """
        Apply ML-based spelling correction using SymSpell.
        
        OPTIMIZED: Only processes lines that appear to need correction.
        Uses fast heuristics to skip clean lines.
        
        With aggressive mode:
        - Processes longer texts (up to 10000 chars)
        - Uses higher edit distance
        - Applies to more lines (max 100)
        """
        if not self.sym_spell:
            return text
        
        # Set limits based on aggressiveness
        max_length = 10000 if aggressive else 5000
        if len(text) > max_length:
            print(f"   ⚠️ Text too long ({len(text)} chars), skipping ML correction")
            return text
        
        try:
            # Process line by line to preserve structure
            lines = text.split('\n')
            corrected_lines = []
            
            # Limit lines to process (SymSpell lookup_compound is slow)
            max_lines = 100 if aggressive else 50
            lines_processed = 0
            
            for line in lines:
                stripped = line.strip()
                
                # Skip conditions (fast checks first)
                if (len(stripped) < 5 or 
                    '__PRESERVED_' in line or
                    lines_processed >= max_lines):
                    corrected_lines.append(line)
                    continue
                
                # Fast heuristic: Skip lines that look clean
                # (mostly alphanumeric, reasonable word length, no weird chars)
                if self._line_looks_clean(stripped):
                    corrected_lines.append(line)
                    continue
                
                lines_processed += 1
                
                # Use compound lookup for context-aware correction
                suggestions = self.sym_spell.lookup_compound(
                    stripped,
                    max_edit_distance=2,
                    ignore_non_words=True,
                    transfer_casing=True
                )
                
                if suggestions:
                    # Preserve original indentation
                    indent = len(line) - len(line.lstrip())
                    corrected_lines.append(' ' * indent + suggestions[0].term)
                else:
                    corrected_lines.append(line)
            
            if lines_processed > 0:
                print(f"   ✓ ML corrected {lines_processed} lines")
            
            return '\n'.join(corrected_lines)
        except Exception as e:
            print(f"   ⚠️ ML correction error: {e}")
            return text
    
    def _line_looks_clean(self, line: str) -> bool:
        """
        Fast heuristic to check if a line looks clean (doesn't need ML correction).
        
        Returns True if line appears to be clean English text.
        """
        if not line:
            return True
        
        # Skip lines with preserved placeholders
        if '__PRESERVED_' in line:
            return True
        
        # Skip mostly numeric lines (dates, amounts, etc.)
        alpha_count = sum(1 for c in line if c.isalpha())
        if alpha_count < len(line) * 0.3:
            return True
        
        # Skip short lines
        if len(line) < 10:
            return True
        
        # Check for OCR-like garbage patterns
        garbage_indicators = [
            # Random consonant clusters
            any(cc in line.lower() for cc in ['xz', 'qx', 'zx', 'jq', 'vx']),
            # Too many special chars
            sum(1 for c in line if not c.isalnum() and c not in ' .,;:\'"-()') > len(line) * 0.15,
            # Single letter words that shouldn't exist
            ' q ' in f' {line.lower()} ' or ' x ' in f' {line.lower()} ' or ' z ' in f' {line.lower()} ',
        ]
        
        # If any garbage indicator found, line needs correction
        if any(garbage_indicators):
            return False
        
        # Line looks clean
        return True
    
    def _final_cleanup(self, text: str) -> str:
        """Final text cleanup."""
        # Fix multiple spaces (but preserve line structure)
        text = re.sub(r' +', ' ', text)
        
        # Fix punctuation spacing
        text = re.sub(r'\s+([.!?,;:])', r'\1', text)
        text = re.sub(r'([.!?,;:])([A-Za-z])', r'\1 \2', text)
        
        # Fix quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Strip each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def correct_legal_document(self, text: str) -> str:
        """
        Specialized correction for legal documents.
        Uses aggressive mode by default but preserves legal terms and structure.
        
        Args:
            text: Legal document text
            
        Returns:
            Corrected text
        """
        # Use aggressive mode for legal documents too (with preservation)
        return self.correct_text(text, aggressive=self.aggressive_mode)
    
    def get_law_explanations(self) -> List[Dict]:
        """
        Get list of detected laws with their explanations.
        Use this to populate popup explanations in the UI.
        
        Returns:
            List of law dictionaries with explanations
        """
        return [law.to_dict() for law in self.detected_laws]
    
    def get_correction_stats(self, original: str, corrected: str) -> Dict:
        """
        Get statistics about corrections made.
        
        Args:
            original: Original text
            corrected: Corrected text
            
        Returns:
            Dict with correction statistics
        """
        original_words = len(original.split())
        corrected_words = len(corrected.split())
        
        # Count character differences
        char_diff = abs(len(original) - len(corrected))
        
        return {
            'original_words': original_words,
            'corrected_words': corrected_words,
            'word_diff': abs(original_words - corrected_words),
            'char_diff': char_diff,
            'reduction_pct': round((1 - len(corrected) / len(original)) * 100, 1) if original else 0
        }


# Convenience function
def correct_ocr_text(text: str, aggressive: bool = True) -> str:
    """
    Quick function to correct OCR text.
    
    Args:
        text: Input text with OCR errors
        aggressive: Use ML-based correction (default: True for better results)
        
    Returns:
        Corrected text
    """
    corrector = OCRTextCorrector(use_ml=True, aggressive_mode=aggressive)
    return corrector.correct_text(text, aggressive=aggressive)


def detect_laws_in_text(text: str) -> List[Dict]:
    """
    Convenience function to detect and explain laws in text.
    
    Args:
        text: Document text
        
    Returns:
        List of law dictionaries with explanations for popup display
    """
    corrector = OCRTextCorrector(use_ml=False)
    corrector.detect_law_references(text)
    return corrector.get_law_explanations()


if __name__ == '__main__':
    # Test with sample OCR text containing law references
    sample_text = """
    ROOM RENTAL AGREEMENT
    
    This is a legally binding agreement pursuant to the Transfer of Property Act, 1882
    and the Indian Contract Act, 1872. It is intended to promote household harmony 
    by clarifying die expectations and responsibilities of die homeowner or Principal 
    Tenant (Landlords) and Tenant when tliey share die same home.
    
    As per Section 106 of the Transfer of Property Act, 1882, the lease period 
    shall be for 11 months.
    
    Tlie term "Landlord" refers to eitlier homeowner or Principal Tenant.
    
    Landlord sliall provide a copy of tliis executed (signed) document to die Tenant, 
    as required by Section 107 of the Registration Act, 1908.
    
    If any pomonyof it is deducted, an accounting will be provided.
    
    Die landlord must give die tenant b&gfs notice of intent to enter.
    
    Contact: +91 98765 43210 or email: landlord@example.com
    Website: www.propertysite.com
    
    This agreement shall be governed by the Karnataka Rent Act, 1999.
    Any disputes shall be settled by arbitration under the Arbitration and Conciliation Act, 1996.
    """
    
    print("=" * 70)
    print("OCR TEXT CORRECTION TEST (with Law Detection)")
    print("=" * 70)
    
    print("\n📄 ORIGINAL TEXT:")
    print("-" * 70)
    print(sample_text)
    
    corrector = OCRTextCorrector(use_ml=True, aggressive_mode=True)
    corrected = corrector.correct_text(sample_text, aggressive=True)
    
    print("\n✅ CORRECTED TEXT:")
    print("-" * 70)
    print(corrected)
    
    stats = corrector.get_correction_stats(sample_text, corrected)
    print("\n📊 CORRECTION STATS:")
    print(f"   Original words: {stats['original_words']}")
    print(f"   Corrected words: {stats['corrected_words']}")
    print(f"   Character difference: {stats['char_diff']}")
    
    print("\n📚 DETECTED LAW REFERENCES (for popup explanations):")
    print("-" * 70)
    laws = corrector.get_law_explanations()
    if laws:
        for i, law in enumerate(laws, 1):
            print(f"\n{i}. {law['law_name']}")
            print(f"   Reference: {law['full_reference']}")
            if law['section']:
                print(f"   Section: {law['section']}")
            print(f"   Explanation: {law['explanation'][:200]}...")
    else:
        print("   No laws detected")
    
    print("\n" + "=" * 70)
