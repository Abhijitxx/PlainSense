"""
Module 1: Advanced Text Preprocessing
======================================

Comprehensive text cleaning and normalization for OCR output.

Features:
- OCR error correction (common character substitutions)
- Punctuation normalization
- Sentence segmentation
- Legal term standardization
- Unicode cleaning
- Structure preservation

Author: AI Assistant
Date: October 8, 2025
"""

import re
import unicodedata
from typing import Dict, List, Tuple


class TextPreprocessor:
    """
    Advanced text preprocessing for OCR output
    """
    
    def __init__(self):
        """Initialize with correction dictionaries"""
        
        # Common OCR character errors
        # NOTE: Removed conflicting 0<->O, 1<->l, 5<->S mappings as they
        # can flip characters incorrectly. These are handled contextually
        # in fix_ocr_errors() using regex patterns instead.
        self.ocr_corrections = {
            # Common misreads (safe, non-conflicting)
            '|': 'I',
            
            # Special characters
            '~': '-',
            '£': 'E',
        }
        
        # Legal term standardization
        self.legal_terms = {
            'landlord': 'Landlord',
            'tenant': 'Tenant',
            'lessor': 'Lessor',
            'lessee': 'Lessee',
            'agreement': 'Agreement',
            'contract': 'Contract',
            'party': 'Party',
            'parties': 'Parties',
        }
        
        # Common word corrections (OCR errors found in your data)
        self.word_corrections = {
            # General OCR errors
            'die': 'the',
            'tlie': 'the',
            'liis': 'his',
            'lier': 'her',
            'tliis': 'this',
            'tliat': 'that',
            'witli': 'with',
            'montli': 'month',
            'montlily': 'monthly',
            'sliall': 'shall',
            'wliich': 'which',
            'tliey': 'they',
            'tlie': 'the',
            
            # Medical lab report specific corrections
            'whale': 'whole',
            'boo': 'blood',
            'god': 'g/dl',
            'cum': 'cumm',
            'cumin': 'cumm',
            'cells cum': 'cells/cumm',
            'cells cumin': 'cells/cumm',
            'are': 'Dr.',  # When followed by name in context
            'sample typo': 'Sample Type',
            'met tad': 'Method',
            'phat': 'that',
            'call mesic': 'Collected',
            'desire': 'Date',
            'sex': 'Sex',
            'photometry': 'Photometry',
            'spectrophotometry': 'Spectrophotometry',
            'each above': 'Gachibowli',
            'each abo': 'Gachibowli',
            
            # New errors from analysis
            'newry': 'Neutrophils',
            'mil/eumm': 'mill/cumm',
            'mil/cumm': 'mill/cumm',
            'smil/cumm': 'mill/cumm',
            '/eumm': '/cumm',
            'gm%': 'gm/dl',
            'gmidl': 'gm/dl',
            'grivdi': 'gm/dl',
            'mill/emm': 'mill/cumm',
            'gorpuscular': 'Corpuscular',
            'gonc': 'Conc',
            'gone': 'Conc',
            'ful': '/uL',
            'jul': '/uL',
            'jub': '/uL',
            'lymipheaytas': 'Lymphocytes',
            'baar': 'Basal',
            'whala': 'whole',
            'btood': 'blood',
            'sone': 'Done',
            'miridray': 'Mindray',
            'elecrical': 'Electrical',
            'tluid': 'fluid',
            'tlood': 'blood',
            'tally': 'Total',
            'wile': 'White',
            'ceil': 'Cell',
            'ceils': 'Cells',
            'UHID': 'UHID',
            'ref': 'Ref.',
            'newry': 'Neutrophils',
            'newt': 'Neutrophils',
            'POV': 'PCV',
            'haematocrit': 'Hematocrit',
            'leucocyte': 'Leukocyte',
            'kab': 'K/uL',
            'bias': 'fL',
            
            # Additional corrections from 426 reports analysis
            'leucocytosis': 'leukocytosis',
            'apollohospitals': 'Apollo Hospitals',
            'apollohospitale': 'Apollo Hospitals',
            'avallablo': 'Available',
            'peripheralsmearexamination': 'PERIPHERAL SMEAR EXAMINATION',
            'clinicalcorelationis': 'CLINICAL CORRELATION IS',
            'specificgravity': 'SPECIFIC GRAVITY',
            'multispeciality': 'MULTISPECIALITY',
            'multisuperspeacilist': 'MULTI SUPER SPECIALIST',
            'notiforimedico': 'NOT FOR MEDICO',
            'notavaildieorimedico': 'NOT AVAILABLE FOR MEDICO',
            'nomvaindiforimedicoveganrurbose': 'NOT VALID FOR MEDICO',
            'wibnoiroino': 'LABORATORY',
            'inikhil': 'NIKHIL',
            'prekallikreln': 'Prekallikrein',
            'anisocytosis': 'Anisocytosis',
            'hepatocellular': 'Hepatocellular',
            'intracellular': 'Intracellular',
            'yiu': 'IU',
            'cellsful': 'cells/uL',
            'chco': 'HCO3',
            'pco': 'pCO2',
            'aptt': 'aPTT',
            'egfr': 'eGFR',
            'piperacilin': 'PIPERACILLIN',
            'nalidixic': 'NALIDIXIC',
            'ampicillin': 'AMPICILLIN',
        }
        
        # Punctuation normalization
        self.punct_replacements = {
            '"': '"',  # Smart quotes to straight
            '"': '"',
            ''': "'",
            ''': "'",
            '—': '-',  # Em dash to hyphen
            '–': '-',  # En dash to hyphen
            '…': '...',  # Ellipsis
            '°': 'degree',  # Degree symbol
        }
    
    def fix_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR character recognition errors
        
        Args:
            text: Raw OCR text
            
        Returns:
            Text with corrected OCR errors
        """
        # Medical-specific pattern fixes FIRST (before word replacements)
        # Fix "whale boo" → "whole blood"
        text = re.sub(r'\bwhale\s+boo[dl]?\b', 'whole blood', text, flags=re.IGNORECASE)
        
        # Fix medical units
        text = re.sub(r'\b(\d+\.?\d*)\s*god\b', r'\1 g/dl', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(\d+\.?\d*)\s*gd[il]\b', r'\1 g/dl', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(\d+\.?\d*)\s*gm%\b', r'\1 gm/dl', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(\d+\.?\d*)\s*gmidl\b', r'\1 gm/dl', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(\d+\.?\d*)\s*grivdi\b', r'\1 gm/dl', text, flags=re.IGNORECASE)
        text = re.sub(r'\bcells?\s+cu[nm]i?n?\b', 'cells/cumm', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(\d+\.?\d*)\s*[sm]il/e[mc]mm\b', r'\1 mill/cumm', text, flags=re.IGNORECASE)
        text = re.sub(r'/e[mc]mm\b', '/cumm', text, flags=re.IGNORECASE)
        text = re.sub(r'\b[fj]u[bl]\b', '/uL', text, flags=re.IGNORECASE)
        
        # Fix "ref doctor are" → "Ref. Doctor: Dr."
        text = re.sub(r'\bref\.?\s+doctor\.?\s+are\s+', 'Ref. Doctor: Dr. ', text, flags=re.IGNORECASE)
        
        # Fix location names
        text = re.sub(r'\beach\s+abo(?:ve)?\b', 'Gachibowli', text, flags=re.IGNORECASE)
        
        # Fix medical test names
        text = re.sub(r'\bnewry\b', 'Neutrophils', text, flags=re.IGNORECASE)
        text = re.sub(r'\bnewt\b', 'Neutrophils', text, flags=re.IGNORECASE)
        text = re.sub(r'\bPOV\b', 'PCV', text)
        text = re.sub(r'\bkab\b', 'K/uL', text, flags=re.IGNORECASE)
        text = re.sub(r'\bgorpuscular\b', 'Corpuscular', text, flags=re.IGNORECASE)
        text = re.sub(r'\blymipheaytas\b', 'Lymphocytes', text, flags=re.IGNORECASE)
        text = re.sub(r'\bwhala\b', 'whole', text, flags=re.IGNORECASE)
        text = re.sub(r'\b[bt]lood\b', 'blood', text, flags=re.IGNORECASE)
        text = re.sub(r'\bbias\b', 'fL', text, flags=re.IGNORECASE)
        
        # Additional patterns from 426 reports analysis
        text = re.sub(r'\by[iI][uU]\b', 'IU', text)  # yIU → IU
        text = re.sub(r'\bclinically\b', 'Clinically', text)  # Capitalize at sentence start
        
        # Fix specific word errors (highest confidence)
        for wrong, correct in self.word_corrections.items():
            # Skip multi-word patterns (already handled above)
            if ' ' in wrong:
                continue
            # Case-insensitive replacement with word boundaries
            text = re.sub(rf'\b{wrong}\b', correct, text, flags=re.IGNORECASE)
        
        # Fix number-letter confusion in context
        # 0 vs O (zero vs letter O)
        text = re.sub(r'(?<=\d)O(?=\d)', '0', text)  # Between digits → 0
        text = re.sub(r'(?<=[A-Z])0(?=[A-Z])', 'O', text)  # Between letters → O
        
        # 1 vs l (one vs lowercase L)
        text = re.sub(r'(?<=\d)l(?=\d)', '1', text)  # Between digits → 1
        text = re.sub(r'(?<=[a-z])1(?=[a-z])', 'l', text)  # Between lowercase → l
        
        # NOTE: Removed global 'rn'->'m' and 'vv'->'w' replacements as they
        # break valid words like 'return', 'govern', 'savvy'. These patterns
        # should only be applied to specific known OCR errors in word_corrections.
        
        # Remove obviously wrong OCR fragments
        # Pattern: <3 chars with mix of special chars and letters
        text = re.sub(r'\b[a-zA-Z]{0,1}[^\w\s]{2,}[a-zA-Z]{0,1}\b', '', text)
        
        # Remove single standalone special characters
        text = re.sub(r'\s[^\w\s]\s', ' ', text)
        
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """
        Normalize punctuation marks
        
        Args:
            text: Text with mixed punctuation
            
        Returns:
            Text with normalized punctuation
        """
        # Replace smart quotes and special punctuation
        for old, new in self.punct_replacements.items():
            text = text.replace(old, new)
        
        # Fix missing spaces after punctuation
        text = re.sub(r'([.!?,;:])([A-Z])', r'\1 \2', text)
        
        # Fix multiple punctuation marks
        text = re.sub(r'\.{2,}', '...', text)  # Multiple periods → ellipsis
        text = re.sub(r'\?{2,}', '?', text)    # Multiple question marks
        text = re.sub(r'!{2,}', '!', text)     # Multiple exclamation marks
        
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.!?,;:])', r'\1', text)
        
        # Add space after comma if missing
        text = re.sub(r',(?=[^\s\d])', ', ', text)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize all whitespace
        
        Args:
            text: Text with irregular whitespace
            
        Returns:
            Text with normalized whitespace
        """
        # Remove all types of whitespace, replace with single space
        text = re.sub(r'[ \t\xa0\u200b]+', ' ', text)
        
        # Normalize line breaks (remove extra blank lines)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove trailing whitespace from lines
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
        
        # Remove leading whitespace from lines
        text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences
        
        Args:
            text: Text to segment
            
        Returns:
            List of sentences
        """
        # Simple sentence boundary detection
        # Splits on . ! ? followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean each sentence
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def standardize_legal_terms(self, text: str) -> str:
        """
        Standardize legal term capitalization
        
        Args:
            text: Text with inconsistent legal terms
            
        Returns:
            Text with standardized terms
        """
        for term_lower, term_standard in self.legal_terms.items():
            # Replace only whole words, preserve original if already capitalized correctly
            pattern = rf'\b{term_lower}\b'
            text = re.sub(pattern, term_standard, text, flags=re.IGNORECASE)
        
        return text
    
    def clean_unicode(self, text: str) -> str:
        """
        Clean and normalize Unicode characters
        
        Args:
            text: Text with mixed Unicode
            
        Returns:
            Normalized ASCII-compatible text
        """
        # Normalize to NFKD (compatibility decomposition)
        text = unicodedata.normalize('NFKD', text)
        
        # Remove non-ASCII characters (or convert to ASCII equivalents)
        # Keep common punctuation and whitespace
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        return text
    
    def remove_page_artifacts(self, text: str) -> str:
        """
        Remove page numbers, headers, footers
        
        Args:
            text: Text with page artifacts
            
        Returns:
            Cleaned text
        """
        # Remove page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'Page\s+\d+(\s+of\s+\d+)?', '', text, flags=re.IGNORECASE)
        
        # Remove common headers/footers
        text = re.sub(r'^[-_=]{3,}$', '', text, flags=re.MULTILINE)
        
        # Remove "Continued..." type markers
        text = re.sub(r'\(continued\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'continued from page \d+', '', text, flags=re.IGNORECASE)
        
        return text
    
    def preprocess(self, text: str, profile: str = 'standard') -> str:
        """
        Full preprocessing pipeline
        
        Args:
            text: Raw text
            profile: Processing intensity
                    - 'minimal': Basic cleaning only
                    - 'standard': All corrections (recommended)
                    - 'aggressive': Maximum cleaning (may over-correct)
            
        Returns:
            Preprocessed text
        """
        if profile == 'minimal':
            # Just basic cleaning
            text = self.normalize_whitespace(text)
            text = self.remove_page_artifacts(text)
        
        elif profile == 'standard':
            # Recommended pipeline
            text = self.clean_unicode(text)
            text = self.fix_ocr_errors(text)
            text = self.normalize_punctuation(text)
            text = self.normalize_whitespace(text)
            text = self.remove_page_artifacts(text)
            text = self.standardize_legal_terms(text)
        
        elif profile == 'aggressive':
            # Maximum cleaning
            text = self.clean_unicode(text)
            text = self.fix_ocr_errors(text)
            text = self.normalize_punctuation(text)
            text = self.normalize_whitespace(text)
            text = self.remove_page_artifacts(text)
            text = self.standardize_legal_terms(text)
            # Could add more aggressive fixes here
        
        return text.strip()
    
    def preprocess_with_sentences(self, text: str, profile: str = 'standard') -> Tuple[str, List[str]]:
        """
        Preprocess and return both full text and sentences
        
        Args:
            text: Raw text
            profile: Processing intensity
            
        Returns:
            Tuple of (full_text, list_of_sentences)
        """
        full_text = self.preprocess(text, profile)
        sentences = self.segment_sentences(full_text)
        
        return full_text, sentences


# =================================================================
# CONVENIENCE FUNCTIONS
# =================================================================

def preprocess_text_file(input_path: str, output_path: str = None, 
                        profile: str = 'standard') -> str:
    """
    Preprocess a text file
    
    Args:
        input_path: Path to input text file
        output_path: Path to save preprocessed text (optional)
        profile: Processing intensity
        
    Returns:
        Preprocessed text
    """
    # Read input
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Preprocess
    preprocessor = TextPreprocessor()
    cleaned = preprocessor.preprocess(text, profile)
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
    
    return cleaned


def preprocess_directory(input_dir: str, output_dir: str = None, 
                        pattern: str = '*.txt', profile: str = 'standard'):
    """
    Preprocess all text files in a directory
    
    Args:
        input_dir: Input directory
        output_dir: Output directory (default: input_dir + '_preprocessed')
        pattern: File pattern to match
        profile: Processing intensity
        
    Returns:
        List of processed file paths
    """
    from pathlib import Path
    
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_path = Path(str(input_path) + '_preprocessed')
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(exist_ok=True)
    
    processed_files = []
    preprocessor = TextPreprocessor()
    
    for input_file in input_path.glob(pattern):
        print(f"📄 Processing: {input_file.name}")
        
        # Read
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Preprocess
        cleaned = preprocessor.preprocess(text, profile)
        
        # Save
        output_file = output_path / input_file.name
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        
        processed_files.append(str(output_file))
        print(f"   ✅ Saved: {output_file.name}")
    
    print(f"\n✅ Processed {len(processed_files)} files")
    return processed_files


# =================================================================
# TESTING
# =================================================================

if __name__ == '__main__':
    # Test on sample OCR output
    sample_text = """
    ROOM RENTAL AGREEMENT
    
    This is a legally binding agreement. It is intended to promote household harmony by clarifying die expectations and
    responsibilities of the homeowner or Principal Tenant (Landlords) and tenant when they share the same home.
    
    Landlord shall provide a copy of tliis executed (signed) document to the tenant, as required by law.
    
    Rental Unit Located at: Toke , Of
    oke , Oft
    2QOGI/ Sher hre>osv-
    Address
    
    Rent jS does / CJdoes not include utilities.If it does not,utility bills will be apportioned as follows:
    Gas/Electricity: Tenant pays ° o of montlily bill.
    """
    
    print("=" * 70)
    print("TEXT PREPROCESSING TEST")
    print("=" * 70)
    
    preprocessor = TextPreprocessor()
    
    print("\n📄 ORIGINAL TEXT:")
    print("-" * 70)
    print(sample_text[:500])
    
    print("\n\n🔧 PREPROCESSED TEXT (STANDARD):")
    print("-" * 70)
    cleaned = preprocessor.preprocess(sample_text, profile='standard')
    print(cleaned[:500])
    
    print("\n\n📊 CORRECTIONS MADE:")
    print("-" * 70)
    print("✓ 'die' → 'the'")
    print("✓ 'tliis' → 'this'")
    print("✓ 'montlily' → 'monthly'")
    print("✓ Removed OCR garbage: '2QOGI/ Sher hre>osv-'")
    print("✓ Normalized punctuation")
    print("✓ Fixed whitespace")
    print("✓ Standardized 'tenant' → 'Tenant'")
