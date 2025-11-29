"""
Clause Segmentation Module
==========================

Segments documents into logical clauses/sections for:
- Legal documents (rental agreements, contracts)
- Medical reports (test sections, findings)

This enables:
- Fine-grained simplification
- Per-clause risk assessment
- Better translation accuracy

Author: PlainSense Team
Date: November 2025
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class DocumentType(Enum):
    """Document type classification"""
    LEGAL = "legal"
    MEDICAL = "medical"
    UNKNOWN = "unknown"


class ClauseType(Enum):
    """Types of clauses/sections"""
    # Legal clause types
    HEADER = "header"
    PARTIES = "parties"
    DEFINITIONS = "definitions"
    TERMS = "terms"
    RENT_PAYMENT = "rent_payment"
    SECURITY_DEPOSIT = "security_deposit"
    MAINTENANCE = "maintenance"
    TERMINATION = "termination"
    PENALTIES = "penalties"
    OBLIGATIONS = "obligations"
    DISPUTE = "dispute"
    SIGNATURES = "signatures"
    
    # Medical section types
    PATIENT_INFO = "patient_info"
    TEST_HEADER = "test_header"
    TEST_RESULTS = "test_results"
    INTERPRETATION = "interpretation"
    RECOMMENDATIONS = "recommendations"
    DOCTOR_INFO = "doctor_info"
    
    # Generic
    GENERAL = "general"
    UNKNOWN = "unknown"


@dataclass
class Clause:
    """Represents a document clause/section"""
    id: int
    clause_type: ClauseType
    title: str
    content: str
    start_line: int
    end_line: int
    risk_level: str  # 'none', 'low', 'medium', 'high', 'critical'
    keywords: List[str]
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['clause_type'] = self.clause_type.value
        return result


class ClauseSegmenter:
    """
    Segments documents into logical clauses/sections
    """
    
    def __init__(self):
        """Initialize with patterns for different document types"""
        
        # Legal document section patterns
        self.legal_section_patterns = [
            # Numbered sections: 1. , 1) , (1) , I. , A.
            (r'^\s*(\d+)\s*[.\)]\s*(.+)', 'numbered'),
            (r'^\s*\((\d+)\)\s*(.+)', 'numbered'),
            (r'^\s*([IVXLC]+)\s*[.\)]\s*(.+)', 'roman'),
            (r'^\s*([A-Z])\s*[.\)]\s*(.+)', 'lettered'),
            
            # Keyword-based sections
            (r'^\s*(WHEREAS|WITNESSETH|NOW THEREFORE)', 'preamble'),
            (r'^\s*(ARTICLE|SECTION|CLAUSE)\s+(\d+|[IVXLC]+)', 'article'),
            (r'^\s*(SCHEDULE|ANNEXURE|APPENDIX)\s*[:\-]?\s*(\w+)?', 'schedule'),
            
            # Common rental agreement headings
            (r'^\s*(RENT(?:AL)?|MONTHLY RENT|PAYMENT)', 'rent'),
            (r'^\s*(SECURITY DEPOSIT|DEPOSIT|CAUTION)', 'deposit'),
            (r'^\s*(TERM(?:INATION)?|DURATION|PERIOD)', 'term'),
            (r'^\s*(MAINTENANCE|REPAIRS|UPKEEP)', 'maintenance'),
            (r'^\s*(NOTICE|TERMINATION|EVICTION)', 'termination'),
            (r'^\s*(PENALTY|PENALTIES|FINE|DAMAGES)', 'penalties'),
            (r'^\s*(OBLIGATION|RESPONSIBILITIES|DUTIES)', 'obligations'),
            (r'^\s*(DISPUTE|ARBITRATION|JURISDICTION)', 'dispute'),
            (r'^\s*(SIGNATURE|SIGNED|WITNESS)', 'signatures'),
        ]
        
        # Medical report section patterns
        self.medical_section_patterns = [
            # Patient information
            (r'^\s*(PATIENT\s*(NAME|INFO|DETAILS)|NAME\s*:|AGE\s*:|SEX\s*:)', 'patient_info'),
            (r'^\s*(SAMPLE|SPECIMEN|COLLECTED)', 'sample_info'),
            
            # Test headers
            (r'^\s*(COMPLETE BLOOD COUNT|CBC|HAEMATOLOGY)', 'cbc'),
            (r'^\s*(BLOOD SUGAR|GLUCOSE|DIABETIC)', 'blood_sugar'),
            (r'^\s*(LIPID PROFILE|CHOLESTEROL)', 'lipid'),
            (r'^\s*(LIVER FUNCTION|LFT|HEPATIC)', 'liver'),
            (r'^\s*(KIDNEY FUNCTION|KFT|RENAL)', 'kidney'),
            (r'^\s*(THYROID|TSH|T3|T4)', 'thyroid'),
            (r'^\s*(URINE|URINALYSIS)', 'urine'),
            (r'^\s*(DIFFERENTIAL COUNT|DLC)', 'differential'),
            
            # Results section
            (r'^\s*(TEST|INVESTIGATION|PARAMETER)\s+.*(RESULT|VALUE)', 'results_header'),
            (r'^\s*(RESULT|FINDINGS|OBSERVATION)', 'results'),
            
            # Interpretation
            (r'^\s*(INTERPRETATION|IMPRESSION|COMMENT|NOTE)', 'interpretation'),
            (r'^\s*(RECOMMENDATION|ADVICE|SUGGESTION)', 'recommendations'),
            
            # Doctor info
            (r'^\s*(DR\.|DOCTOR|PATHOLOGIST|REPORTED BY)', 'doctor'),
        ]
        
        # Risk keywords by category
        self.legal_risk_keywords = {
            'high': [
                'penalty', 'penalties', 'fine', 'fines', 'damages',
                'forfeit', 'forfeiture', 'eviction', 'evict',
                'terminate immediately', 'breach', 'violation',
                'liable', 'liability', 'indemnify', 'indemnification',
                'legal action', 'court', 'lawsuit', 'sue',
            ],
            'medium': [
                'notice period', 'lock-in', 'lock in', 'minimum period',
                'security deposit', 'caution deposit', 'advance',
                'increase', 'revision', 'escalation',
                'restriction', 'prohibited', 'not allowed',
                'obligation', 'must', 'shall', 'required',
            ],
            'low': [
                'maintenance', 'repair', 'upkeep',
                'electricity', 'water', 'utility',
                'inspection', 'access',
            ]
        }
        
        self.medical_risk_keywords = {
            'critical': [
                'critical', 'panic', 'emergency', 'urgent',
                'immediately', 'life-threatening',
            ],
            'high': [
                'abnormal', 'high', 'low', 'elevated', 'decreased',
                'positive', 'detected', 'present',
            ],
            'medium': [
                'borderline', 'slightly', 'mild', 'moderate',
            ],
        }
    
    def detect_document_type(self, text: str) -> DocumentType:
        """
        Detect if document is legal or medical
        
        Args:
            text: Document text
            
        Returns:
            DocumentType enum
        """
        text_lower = text.lower()
        
        # Legal document indicators
        legal_indicators = [
            'agreement', 'contract', 'landlord', 'tenant', 'lessor', 'lessee',
            'hereby', 'whereas', 'witnesseth', 'party', 'parties',
            'rent', 'lease', 'rental', 'premises', 'property',
            'term', 'termination', 'notice period', 'security deposit',
        ]
        
        # Medical document indicators
        medical_indicators = [
            'patient', 'test', 'result', 'blood', 'sample', 'specimen',
            'hemoglobin', 'glucose', 'cholesterol', 'mg/dl', 'g/dl',
            'reference range', 'normal', 'abnormal', 'laboratory',
            'pathologist', 'doctor', 'hospital', 'clinic', 'lab',
            'cells/cumm', 'million/cumm', 'report',
        ]
        
        legal_score = sum(1 for ind in legal_indicators if ind in text_lower)
        medical_score = sum(1 for ind in medical_indicators if ind in text_lower)
        
        if medical_score > legal_score + 3:
            return DocumentType.MEDICAL
        elif legal_score > medical_score + 3:
            return DocumentType.LEGAL
        elif medical_score > legal_score:
            return DocumentType.MEDICAL
        elif legal_score > medical_score:
            return DocumentType.LEGAL
        else:
            return DocumentType.UNKNOWN
    
    def _find_section_boundaries(self, lines: List[str], 
                                  doc_type: DocumentType) -> List[Tuple[int, str, str]]:
        """
        Find section boundaries in document
        
        Returns:
            List of (line_number, section_type, title)
        """
        boundaries = []
        patterns = (self.medical_section_patterns if doc_type == DocumentType.MEDICAL 
                   else self.legal_section_patterns)
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check against patterns
            for pattern, section_type in patterns:
                match = re.match(pattern, line_stripped, re.IGNORECASE)
                if match:
                    boundaries.append((i, section_type, line_stripped))
                    break
            
            # Also detect by formatting (all caps lines often indicate headers)
            if (line_stripped.isupper() and 
                len(line_stripped) > 5 and 
                len(line_stripped) < 100 and
                not any(char.isdigit() for char in line_stripped[:5])):
                # Likely a header
                if not any(b[0] == i for b in boundaries):
                    boundaries.append((i, 'header', line_stripped))
        
        return boundaries
    
    def _calculate_risk_level(self, content: str, doc_type: DocumentType) -> Tuple[str, List[str]]:
        """
        Calculate risk level for a clause
        
        Returns:
            Tuple of (risk_level, list_of_keywords_found)
        """
        content_lower = content.lower()
        keywords_found = []
        
        risk_keywords = (self.medical_risk_keywords if doc_type == DocumentType.MEDICAL 
                        else self.legal_risk_keywords)
        
        # Check each risk level
        for level in ['critical', 'high', 'medium', 'low']:
            if level in risk_keywords:
                for keyword in risk_keywords[level]:
                    if keyword in content_lower:
                        keywords_found.append(keyword)
                        if level in ['critical', 'high']:
                            return (level, keywords_found)
        
        if keywords_found:
            # Return highest risk level found
            for level in ['critical', 'high', 'medium', 'low']:
                if level in risk_keywords:
                    for keyword in risk_keywords[level]:
                        if keyword in keywords_found:
                            return (level, keywords_found)
        
        return ('none', [])
    
    def _map_section_to_clause_type(self, section_type: str, 
                                     doc_type: DocumentType) -> ClauseType:
        """Map section type string to ClauseType enum"""
        
        if doc_type == DocumentType.MEDICAL:
            mapping = {
                'patient_info': ClauseType.PATIENT_INFO,
                'sample_info': ClauseType.PATIENT_INFO,
                'cbc': ClauseType.TEST_HEADER,
                'blood_sugar': ClauseType.TEST_HEADER,
                'lipid': ClauseType.TEST_HEADER,
                'liver': ClauseType.TEST_HEADER,
                'kidney': ClauseType.TEST_HEADER,
                'thyroid': ClauseType.TEST_HEADER,
                'urine': ClauseType.TEST_HEADER,
                'differential': ClauseType.TEST_HEADER,
                'results_header': ClauseType.TEST_RESULTS,
                'results': ClauseType.TEST_RESULTS,
                'interpretation': ClauseType.INTERPRETATION,
                'recommendations': ClauseType.RECOMMENDATIONS,
                'doctor': ClauseType.DOCTOR_INFO,
            }
        else:  # Legal
            mapping = {
                'numbered': ClauseType.TERMS,
                'roman': ClauseType.TERMS,
                'lettered': ClauseType.TERMS,
                'preamble': ClauseType.HEADER,
                'article': ClauseType.TERMS,
                'schedule': ClauseType.TERMS,
                'rent': ClauseType.RENT_PAYMENT,
                'deposit': ClauseType.SECURITY_DEPOSIT,
                'term': ClauseType.TERMINATION,
                'maintenance': ClauseType.MAINTENANCE,
                'termination': ClauseType.TERMINATION,
                'penalties': ClauseType.PENALTIES,
                'obligations': ClauseType.OBLIGATIONS,
                'dispute': ClauseType.DISPUTE,
                'signatures': ClauseType.SIGNATURES,
                'header': ClauseType.HEADER,
            }
        
        return mapping.get(section_type, ClauseType.GENERAL)
    
    def segment(self, text: str, doc_type: Optional[DocumentType] = None) -> Dict:
        """
        Segment document into clauses
        
        Args:
            text: Document text
            doc_type: Optional document type (auto-detected if not provided)
                      Can be DocumentType enum, or string 'legal'/'medical'
            
        Returns:
            Dictionary with clauses and metadata
        """
        # Convert string doc_type to DocumentType enum if needed
        if isinstance(doc_type, str):
            doc_type = DocumentType.LEGAL if doc_type.lower() == 'legal' else DocumentType.MEDICAL
        
        # Detect document type if not provided
        if doc_type is None:
            doc_type = self.detect_document_type(text)
        
        lines = text.split('\n')
        
        # Find section boundaries
        boundaries = self._find_section_boundaries(lines, doc_type)
        
        # If no boundaries found, segment by paragraphs
        if not boundaries:
            return self._segment_by_paragraphs(text, doc_type)
        
        # Create clauses from boundaries
        clauses = []
        
        for i, (line_num, section_type, title) in enumerate(boundaries):
            # Determine end line
            if i + 1 < len(boundaries):
                end_line = boundaries[i + 1][0] - 1
            else:
                end_line = len(lines) - 1
            
            # Extract content
            content_lines = lines[line_num:end_line + 1]
            content = '\n'.join(content_lines).strip()
            
            # If content is very long (>500 chars), split by sentences for better granularity
            if len(content) > 500:
                # Use sentence-based segmentation for this large section
                sub_result = self._segment_by_logical_clauses(content, doc_type)
                for sub_clause in sub_result.get('clauses', []):
                    # Skip filler content
                    if self._is_filler_clause(sub_clause['content']):
                        continue
                    clauses.append(sub_clause)
                continue
            
            # Skip filler clauses
            if self._is_filler_clause(content):
                continue
            
            # Calculate risk
            risk_level, keywords = self._calculate_risk_level(content, doc_type)
            
            clause = Clause(
                id=i + 1,
                clause_type=self._map_section_to_clause_type(section_type, doc_type),
                title=title[:100],  # Limit title length
                content=content,
                start_line=line_num + 1,
                end_line=end_line + 1,
                risk_level=risk_level,
                keywords=keywords
            )
            
            clauses.append(clause)
        
        # Renumber clauses sequentially
        for idx, clause in enumerate(clauses):
            if isinstance(clause, dict):
                clause['id'] = idx + 1
            elif isinstance(clause, Clause):
                clause.id = idx + 1
        
        # Generate summary
        high_risk = [c for c in clauses if (isinstance(c, dict) and c.get('risk_level') in ['high', 'critical']) or 
                     (isinstance(c, Clause) and c.risk_level in ['high', 'critical'])]
        
        # Convert all to dicts for consistent output
        clause_dicts = []
        for c in clauses:
            if isinstance(c, dict):
                clause_dicts.append(c)
            elif isinstance(c, Clause):
                clause_dicts.append(c.to_dict())
            else:
                clause_dicts.append(c)
        
        return {
            'document_type': doc_type.value,
            'total_clauses': len(clause_dicts),
            'high_risk_clauses': len(high_risk),
            'clauses': clause_dicts,
            'high_risk_summary': [
                {'id': c.get('id', i), 'title': c.get('title', ''), 'risk': c.get('risk_level', 'none'), 'keywords': c.get('keywords', [])}
                for i, c in enumerate(clause_dicts) if c.get('risk_level') in ['high', 'critical']
            ],
            'segmentation_method': 'hybrid'
        }
    
    def _is_filler_clause(self, text: str) -> bool:
        """
        Check if a clause is a filler (witness blocks, signatures, headers without content).
        These don't provide meaningful information and should be filtered out.
        """
        text_lower = text.lower().strip()
        text_clean = re.sub(r'[\s\n]+', ' ', text_lower)  # Normalize whitespace
        
        # Too short to be meaningful (less than 50 chars after cleanup)
        if len(text_clean) < 50:
            return True
        
        # Count actual words (ignore punctuation and numbers)
        actual_words = [w for w in re.findall(r'[a-zA-Z]+', text_lower) if len(w) > 2]
        if len(actual_words) < 8:
            return True
        
        # Filler patterns to skip - now checking against cleaned text too
        filler_patterns = [
            # Witness blocks
            r'witness[es]*[:\s]*$',
            r'in\s+witness\s+whereof',
            r'as\s+witness',
            r'witnessed\s+by',
            r'before\s+me.*witness',
            
            # Signature blocks
            r'signed?[:\s]*$',
            r'signature[s]?[:\s]*$',
            r'sign(ed|ature)?\s*(by|of)?\s*(landlord|tenant|lessor|lessee|party)',
            r'executed\s+(at|on|by)',
            r'duly\s+executed',
            
            # Party labels
            r'^\s*(first|second|1st|2nd|third|3rd)\s+party\s*:?',
            r'^\s*(landlord|tenant|lessor|lessee|owner|occupant)\s*:?\s*$',
            r'^\s*party\s+(of\s+the\s+)?(first|second)\s+part',
            
            # Empty labels
            r'^\s*name[:\s]*$',
            r'^\s*address[:\s]*$',
            r'^\s*phone[:\s]*$',
            r'^\s*date[:\s]*$',
            r'^\s*place[:\s]*$',
            r'^\s*seal[:\s]*$',
            
            # Placeholders
            r'^\s*\d+\.?\s*$',
            r'^\s*[a-z]\)?\s*$',
            r'^\s*_+\s*$',
            r'^\s*\.+\s*$',
            r'^\s*\(\s*\)\s*$',
            r'^\s*\[\s*\]\s*$',
            
            # Common non-content phrases
            r'^\s*(schedule|annexure|appendix)\s*[a-z0-9]*\s*$',
            r'^\s*for\s+and\s+on\s+behalf\s+of',
            r'^\s*signed\s+sealed\s+and\s+delivered',
            r'^\s*in\s+the\s+presence\s+of',
        ]
        
        for pattern in filler_patterns:
            if re.search(pattern, text_clean, re.IGNORECASE):
                return True
        
        # Check if content is mostly just labels/names without substantive content
        # Ratio of label words to total words
        label_words = {
            'witness', 'witnesses', 'landlord', 'tenant', 'lessor', 'lessee', 
            'party', 'parties', 'signed', 'signature', 'name', 'date', 'place',
            'address', 'phone', 'seal', 'executed', 'presence', 'behalf',
            'first', 'second', 'third', 'owner', 'occupant', 'mr', 'mrs', 'ms',
            'sri', 'smt', 'shri', 'dr', 'prop', 'proprietor'
        }
        label_count = sum(1 for w in actual_words if w in label_words)
        if len(actual_words) > 0 and label_count / len(actual_words) > 0.5:
            return True
        
        # Check for sentences with actual verbs (meaningful content)
        verb_patterns = [
            r'\b(shall|will|must|may|should|can|would|could)\b',
            r'\b(agree|pay|maintain|provide|ensure|comply|notify|terminate|vacate|return)\b',
            r'\b(is|are|was|were|has|have|had)\b.*\b(responsible|liable|required|entitled|obligated)\b',
        ]
        has_verb = any(re.search(p, text_lower) for p in verb_patterns)
        if not has_verb and len(actual_words) < 20:
            return True
        
        return False
    
    def _segment_by_paragraphs(self, text: str, doc_type: DocumentType) -> Dict:
        """Fallback: segment by paragraphs when no clear sections found"""
        
        # Split by double newlines or multiple newlines
        paragraphs = re.split(r'\n\s*\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # If we only have 1-2 paragraphs and text is long, use sentence-based segmentation
        if len(paragraphs) <= 2 and len(text) > 500:
            return self._segment_by_logical_clauses(text, doc_type)
        
        clauses = []
        current_line = 1
        clause_id = 0
        
        for i, para in enumerate(paragraphs):
            para_lines = para.count('\n') + 1
            
            # Skip filler clauses
            if self._is_filler_clause(para):
                current_line += para_lines + 1
                continue
            
            clause_id += 1
            
            # Calculate risk
            risk_level, keywords = self._calculate_risk_level(para, doc_type)
            
            # Try to extract title from first line
            first_line = para.split('\n')[0][:80]
            
            clause = Clause(
                id=clause_id,
                clause_type=ClauseType.GENERAL,
                title=first_line,
                content=para,
                start_line=current_line,
                end_line=current_line + para_lines - 1,
                risk_level=risk_level,
                keywords=keywords
            )
            
            clauses.append(clause)
            current_line += para_lines + 1  # +1 for blank line
        
        high_risk = [c for c in clauses if c.risk_level in ['high', 'critical']]
        
        return {
            'document_type': doc_type.value,
            'total_clauses': len(clauses),
            'high_risk_clauses': len(high_risk),
            'clauses': [c.to_dict() for c in clauses],
            'high_risk_summary': [
                {'id': c.id, 'title': c.title, 'risk': c.risk_level, 'keywords': c.keywords}
                for c in high_risk
            ],
            'segmentation_method': 'paragraph'
        }
    
    def _segment_by_logical_clauses(self, text: str, doc_type: DocumentType) -> Dict:
        """
        Segment dense legal text by logical clause boundaries.
        Used when text is continuous without paragraph breaks.
        
        Each sentence that expresses a distinct obligation, right, or condition
        becomes its own clause for better analysis.
        """
        
        # First, split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 30]
        
        clauses = []
        clause_id = 0
        
        for segment in sentences:
            # Skip filler clauses
            if self._is_filler_clause(segment):
                continue
            
            clause_id += 1
            
            # Calculate risk
            risk_level, keywords = self._calculate_risk_level(segment, doc_type)
            
            # Generate title from content
            title = self._generate_clause_title(segment)
            
            clause = Clause(
                id=clause_id,
                clause_type=self._detect_clause_type(segment),
                title=title,
                content=segment,
                start_line=1,
                end_line=1,
                risk_level=risk_level,
                keywords=keywords
            )
            
            clauses.append(clause)
        
        high_risk = [c for c in clauses if c.risk_level in ['high', 'critical']]
        
        return {
            'document_type': doc_type.value,
            'total_clauses': len(clauses),
            'high_risk_clauses': len(high_risk),
            'clauses': [c.to_dict() for c in clauses],
            'high_risk_summary': [
                {'id': c.id, 'title': c.title, 'risk': c.risk_level, 'keywords': c.keywords}
                for c in high_risk
            ],
            'segmentation_method': 'logical_clause'
        }
    
    def _split_by_sentence_groups(self, text: str) -> List[str]:
        """
        Split text into individual clauses (1-2 sentences each).
        Each sentence that introduces a new obligation/right becomes its own clause.
        """
        # Split by sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        
        if len(sentences) <= 2:
            return sentences
        
        # Each sentence that starts a new topic becomes its own clause
        # Only group continuation sentences with their parent
        groups = []
        
        # Keywords that indicate a standalone clause
        standalone_indicators = [
            r'^the\s+(landlord|tenant|lessor|lessee)',
            r'^this\s+(agreement|rental|lease)',
            r'^if\s+',
            r'^in\s+(case|the\s+event)',
            r'^at\s+the\s+(termination|end)',
            r'^upon\s+',
            r'^should\s+',
            r'^neither\s+',
            r'^either\s+',
            r'^both\s+',
            r'^any\s+',
            r'^no\s+(party|person)',
            r'^it\s+is\s+(agreed|hereby)',
        ]
        
        def is_standalone(sentence):
            """Check if sentence should be its own clause"""
            s_lower = sentence.lower().strip()
            for pattern in standalone_indicators:
                if re.match(pattern, s_lower):
                    return True
            return False
        
        for sentence in sentences:
            # Each sentence becomes its own clause
            # This gives more granular analysis
            groups.append(sentence)
        
        return groups
    
    def _generate_clause_title(self, content: str) -> str:
        """Generate a descriptive title for a clause based on its content"""
        content_lower = content.lower()
        
        # Map content to descriptive titles
        title_patterns = [
            (r'\b(rent|monthly\s+rent|payment.*landlord)\b', 'Rent Payment Terms'),
            (r'\b(security\s+deposit|advance|caution\s+deposit)\b', 'Security Deposit'),
            (r'\bperiod\s+of\s+(this\s+)?(agreement|tenancy)\b', 'Tenancy Period'),
            (r'\b(terminat|vacat|surrender)\b', 'Termination & Vacancy'),
            (r'\b(notice|three\s+months?\s+notice)\b', 'Notice Period'),
            (r'\bpro[\-\s]?rat', 'Pro-rated Rent'),
            (r'\b(tax|municipal|assessment|rates)\b', 'Taxes & Assessments'),
            (r'\b(electric|water|utility)\b', 'Utilities & Charges'),
            (r'\b(condition|wear\s+and\s+tear|maintenance)\b', 'Property Condition & Maintenance'),
            (r'\b(sublet|let\s+out|assign)\b', 'Subletting Restrictions'),
            (r'\b(waste|damage)\b', 'Property Care'),
            (r'\bagree[ds]?\s+and\s+declare[ds]?\b', 'Agreement Declaration'),
        ]
        
        for pattern, title in title_patterns:
            if re.search(pattern, content_lower):
                return title
        
        # Default: use first few words
        words = content.split()[:6]
        return ' '.join(words) + '...' if len(words) == 6 else ' '.join(words)
    
    def _detect_clause_type(self, content: str) -> ClauseType:
        """Detect the clause type based on content"""
        content_lower = content.lower()
        
        if re.search(r'\b(rent|monthly\s+rent|payment.*landlord|rupees|rs\.?)\b', content_lower):
            return ClauseType.RENT_PAYMENT
        elif re.search(r'\b(security\s+deposit|advance|caution)\b', content_lower):
            return ClauseType.SECURITY_DEPOSIT
        elif re.search(r'\b(terminat|notice\s+period|vacat|evict)\b', content_lower):
            return ClauseType.TERMINATION
        elif re.search(r'\b(maintain|repair|upkeep|condition)\b', content_lower):
            return ClauseType.MAINTENANCE
        elif re.search(r'\b(penalty|fine|forfeit|damages)\b', content_lower):
            return ClauseType.PENALTIES
        elif re.search(r'\b(obligation|shall|must|agree)\b', content_lower):
            return ClauseType.OBLIGATIONS
        elif re.search(r'\b(dispute|arbitrat|jurisdiction|court)\b', content_lower):
            return ClauseType.DISPUTE
        else:
            return ClauseType.GENERAL
    
    def segment_medical_by_tests(self, text: str) -> Dict:
        """
        Special segmentation for medical reports: one segment per test
        
        Args:
            text: Medical report OCR text
            
        Returns:
            Segmented by individual tests
        """
        # Import medical parser for test extraction
        try:
            from medical_report_parser import MedicalReportParser
            parser = MedicalReportParser()
            parsed = parser.parse_report(text)
            
            clauses = []
            
            # Create clause for each test result
            for i, result in enumerate(parsed['results'], 1):
                risk = 'none'
                if result['flag'] in ['critical_low', 'critical_high']:
                    risk = 'critical'
                elif result['flag'] in ['low', 'high']:
                    risk = 'high'
                
                clause = Clause(
                    id=i,
                    clause_type=ClauseType.TEST_RESULTS,
                    title=result['test_name_standardized'],
                    content=result['raw_text'],
                    start_line=result['line_number'],
                    end_line=result['line_number'],
                    risk_level=risk,
                    keywords=[result['flag']] if result['flag'] != 'normal' else []
                )
                clauses.append(clause)
            
            return {
                'document_type': 'medical',
                'total_clauses': len(clauses),
                'high_risk_clauses': sum(1 for c in clauses if c.risk_level in ['high', 'critical']),
                'clauses': [c.to_dict() for c in clauses],
                'parsed_results': parsed['results'],
                'summary': parsed['summary']
            }
            
        except ImportError:
            # Fall back to standard segmentation
            return self.segment(text, DocumentType.MEDICAL)


# =============================================================================
# Convenience Functions
# =============================================================================

def segment_document(text: str) -> Dict:
    """
    Quick function to segment any document
    
    Args:
        text: Document text
        
    Returns:
        Segmented clauses
    """
    segmenter = ClauseSegmenter()
    return segmenter.segment(text)


def segment_legal_document(text: str) -> Dict:
    """Segment a legal document"""
    segmenter = ClauseSegmenter()
    return segmenter.segment(text, DocumentType.LEGAL)


def segment_medical_report(text: str) -> Dict:
    """Segment a medical report by tests"""
    segmenter = ClauseSegmenter()
    return segmenter.segment_medical_by_tests(text)


def detect_type(text: str) -> str:
    """Detect document type"""
    segmenter = ClauseSegmenter()
    return segmenter.detect_document_type(text).value


# =============================================================================
# Main - Demo
# =============================================================================

if __name__ == '__main__':
    # Sample rental agreement
    legal_sample = """
    RENTAL AGREEMENT
    
    This Rental Agreement is made on 28th November 2025 between:
    
    LANDLORD: Mr. John Smith, residing at 123 Main Street, Bangalore
    TENANT: Mr. Raj Kumar, residing at 456 Park Avenue, Chennai
    
    WHEREAS the Landlord is the owner of the property located at 789 Lake View Road.
    
    1. RENT AND PAYMENT
    The monthly rent shall be Rs. 25,000 (Twenty-Five Thousand Rupees).
    Payment is due on the 5th of every month.
    Late payment will attract a penalty of Rs. 500 per day.
    
    2. SECURITY DEPOSIT
    The Tenant shall pay a security deposit of Rs. 75,000.
    This deposit will be refunded within 30 days of vacating the premises.
    Deductions may be made for any damages to the property.
    
    3. TERM AND TERMINATION
    The agreement is for a period of 11 months.
    Lock-in period: First 6 months, neither party can terminate.
    Notice period: 2 months written notice required for termination.
    
    4. MAINTENANCE
    Minor repairs up to Rs. 2,000 to be borne by the Tenant.
    Major repairs and structural maintenance by the Landlord.
    
    5. PENALTIES AND BREACH
    Breach of any clause may result in immediate termination.
    The Landlord reserves the right to evict the Tenant with 7 days notice.
    The Tenant shall be liable for any legal costs incurred.
    
    SIGNED:
    Landlord: _____________
    Tenant: _____________
    """
    
    print("="*70)
    print("CLAUSE SEGMENTATION - DEMO")
    print("="*70)
    
    segmenter = ClauseSegmenter()
    
    # Test document type detection
    doc_type = segmenter.detect_document_type(legal_sample)
    print(f"\n📄 Detected document type: {doc_type.value.upper()}")
    
    # Segment the document
    result = segmenter.segment(legal_sample)
    
    print(f"\n📊 Total clauses found: {result['total_clauses']}")
    print(f"⚠️  High-risk clauses: {result['high_risk_clauses']}")
    
    print("\n" + "="*70)
    print("CLAUSE BREAKDOWN:")
    print("="*70)
    
    for clause in result['clauses']:
        risk_icon = "🔴" if clause['risk_level'] in ['high', 'critical'] else "🟡" if clause['risk_level'] == 'medium' else "🟢"
        print(f"\n{risk_icon} Clause {clause['id']}: {clause['title'][:50]}...")
        print(f"   Type: {clause['clause_type']}")
        print(f"   Risk: {clause['risk_level']}")
        if clause['keywords']:
            print(f"   Keywords: {', '.join(clause['keywords'][:5])}")
    
    print("\n" + "="*70)
    print("HIGH-RISK SUMMARY:")
    print("="*70)
    
    for item in result['high_risk_summary']:
        print(f"\n🚨 Clause {item['id']}: {item['title'][:40]}...")
        print(f"   Risk keywords: {', '.join(item['keywords'])}")
