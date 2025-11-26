"""
ML-Based Text Preprocessing with SymSpell
==========================================

SymSpell is PERFECT for your use case because:
- Very fast (1 million words/sec)
- Excellent for OCR errors
- No GPU needed
- Easy to integrate

Installation:
    pip install symspellpy

Usage:
    python ml_preprocessing_symspell.py
"""

import re
from pathlib import Path

try:
    from symspellpy import SymSpell, Verbosity
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False
    print("⚠️  SymSpell not installed. Run: pip install symspellpy")


class MLTextPreprocessor:
    """
    ML-powered text preprocessor using SymSpell
    """
    
    def __init__(self):
        """Initialize SymSpell"""
        if not SYMSPELL_AVAILABLE:
            raise ImportError("SymSpell not installed")
        
        # Create SymSpell instance
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        
        # Load dictionary (comes with symspellpy)
        dictionary_path = self._get_dictionary_path()
        
        if dictionary_path and Path(dictionary_path).exists():
            self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
            print(f"✅ Loaded dictionary: {len(self.sym_spell.words)} words")
        else:
            print("⚠️  Dictionary not found. Using basic correction.")
    
    def _get_dictionary_path(self):
        """Get path to frequency dictionary"""
        # SymSpell comes with a dictionary
        try:
            import pkg_resources
            return pkg_resources.resource_filename(
                'symspellpy',
                'frequency_dictionary_en_82_765.txt'
            )
        except:
            # Manual path if needed
            return "frequency_dictionary_en_82_765.txt"
    
    def correct_word(self, word):
        """Correct a single word"""
        suggestions = self.sym_spell.lookup(
            word,
            Verbosity.CLOSEST,
            max_edit_distance=2
        )
        
        if suggestions:
            return suggestions[0].term
        return word
    
    def correct_text(self, text):
        """Correct entire text with context awareness"""
        # Use compound correction (considers word context)
        suggestions = self.sym_spell.lookup_compound(
            text,
            max_edit_distance=2,
            ignore_non_words=True
        )
        
        if suggestions:
            return suggestions[0].term
        return text
    
    def preprocess(self, text, use_compound=True):
        """
        Full preprocessing with ML correction
        
        Args:
            text: Raw text
            use_compound: Use context-aware correction (recommended)
        
        Returns:
            Corrected text
        """
        # Basic cleaning first
        text = self._basic_cleanup(text)
        
        # ML-based correction
        if use_compound:
            # Context-aware (better for sentences)
            text = self.correct_text(text)
        else:
            # Word-by-word (faster)
            words = text.split()
            corrected_words = [self.correct_word(w) for w in words]
            text = ' '.join(corrected_words)
        
        # Final cleanup
        text = self._final_cleanup(text)
        
        return text
    
    def _basic_cleanup(self, text):
        """Basic text cleaning before ML"""
        # Remove excessive whitespace
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    def _final_cleanup(self, text):
        """Final cleanup after ML correction"""
        # Fix punctuation spacing
        text = re.sub(r'\s+([.!?,;:])', r'\1', text)
        text = re.sub(r'([.!?,;:])([A-Za-z])', r'\1 \2', text)
        
        return text.strip()


def compare_corrections(text):
    """Compare hard-coded vs ML-based correction"""
    print("\n" + "="*70)
    print("COMPARISON: Hard-coded vs ML-based")
    print("="*70)
    
    print("\n📄 Original Text:")
    print("-"*70)
    print(text)
    
    # Hard-coded approach
    from module1_text_preprocessing import TextPreprocessor
    hard_coded = TextPreprocessor()
    result1 = hard_coded.preprocess(text)
    
    print("\n🔧 Hard-coded Corrections:")
    print("-"*70)
    print(result1)
    
    # ML approach
    if SYMSPELL_AVAILABLE:
        ml_based = MLTextPreprocessor()
        result2 = ml_based.preprocess(text)
        
        print("\n🤖 ML-based Corrections (SymSpell):")
        print("-"*70)
        print(result2)
    else:
        print("\n⚠️  Install SymSpell to see ML-based corrections:")
        print("   pip install symspellpy")


def process_file_with_ml(input_file, output_file=None):
    """Process a file using ML correction"""
    if not SYMSPELL_AVAILABLE:
        print("❌ SymSpell not installed. Run: pip install symspellpy")
        return
    
    # Read file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"\n📄 Processing: {input_file}")
    print(f"   Original: {len(text.split())} words")
    
    # Process
    preprocessor = MLTextPreprocessor()
    cleaned = preprocessor.preprocess(text)
    
    print(f"   Corrected: {len(cleaned.split())} words")
    
    # Save
    if output_file is None:
        output_file = str(input_file).replace('.txt', '_ml_corrected.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    
    print(f"   ✅ Saved to: {output_file}")


if __name__ == '__main__':
    # Test with sample text
    sample_text = """
    RENTAL AGREEMENT
    
    Tlie landlord agrees to let out die premises for montlily rent of Rs. 15000.
    
    The tenant sliall pay on 5tli of every montli. The agreement is valid for
    twelve montlis.
    
    The landlord will retain tlie original.
    """
    
    print("\n" + "="*70)
    print("ML-BASED TEXT PREPROCESSING WITH SYMSPELL")
    print("="*70)
    
    if SYMSPELL_AVAILABLE:
        # Compare approaches
        compare_corrections(sample_text)
        
        print("\n" + "="*70)
        print("INTEGRATION GUIDE")
        print("="*70)
        print("\n1. ✅ SymSpell installed and working!")
        print("\n2. To use in your pipeline:")
        print("   from ml_preprocessing_symspell import MLTextPreprocessor")
        print("   preprocessor = MLTextPreprocessor()")
        print("   cleaned = preprocessor.preprocess(your_text)")
        print("\n3. To process files:")
        print("   process_file_with_ml('input.txt', 'output.txt')")
    else:
        print("\n⚠️  Install SymSpell first:")
        print("\n   pip install symspellpy")
        print("\nThen run this script again to see ML corrections!")
