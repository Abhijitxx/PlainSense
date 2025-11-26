"""
COMPLETE OCR PIPELINE - Production Ready
=========================================

This is the master pipeline that integrates everything:
1. Document input (DOCX, PDF, Images)
2. Smart routing (Direct extraction vs OCR)
3. Image preprocessing (for scanned docs)
4. OCR extraction (Tesseract)
5. ML-based text correction (SymSpell)
6. Text normalization
7. Output management

Author: AI Assistant
Date: November 26, 2025
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import time

# Import all modules
from document_extractor import CompleteDocumentExtractor
from image_preprocessor import ImagePreprocessor
from text_preprocessor import TextPreprocessor

# Try to import ML preprocessor
try:
    from ml_text_corrector import MLTextPreprocessor
    ML_AVAILABLE = True
except (ImportError, Exception) as e:
    ML_AVAILABLE = False
    print(f"⚠️  ML preprocessing not available: {e}")
    print("   Install: pip install symspellpy")


class CompletePipeline:
    """
    Production-ready OCR pipeline with all features integrated
    """
    
    def __init__(
        self,
        output_dir: str = "output",
        use_ml_correction: bool = True,
        ml_profile: str = "symspell"  # 'symspell', 'hard-coded', or 'both'
    ):
        """
        Initialize the complete pipeline
        
        Args:
            output_dir: Base output directory
            use_ml_correction: Use ML-based text correction
            ml_profile: Which correction method to use
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.final_output = self.output_dir / "final_preprocessed"
        self.final_output.mkdir(exist_ok=True)
        
        # Initialize components
        print("\n" + "="*70)
        print("INITIALIZING COMPLETE OCR PIPELINE")
        print("="*70)
        
        print("\n1️⃣  Loading Document Extractor...")
        self.extractor = CompleteDocumentExtractor(output_base_dir=self.output_dir)
        print("   ✅ Document extractor ready (Direct + OCR)")
        
        print("\n2️⃣  Loading Image Preprocessor...")
        self.image_preprocessor = ImagePreprocessor()
        print("   ✅ Image preprocessor ready (Deskew, Denoise, Threshold)")
        
        print("\n3️⃣  Loading Text Preprocessor...")
        self.text_preprocessor = TextPreprocessor()
        print("   ✅ Text preprocessor ready (13 hard-coded corrections)")
        
        # ML Preprocessor
        self.use_ml = use_ml_correction and ML_AVAILABLE
        self.ml_profile = ml_profile
        
        if self.use_ml:
            print("\n4️⃣  Loading ML Text Corrector (SymSpell)...")
            try:
                self.ml_preprocessor = MLTextPreprocessor()
                print("   ✅ ML preprocessor ready (82,765 words dictionary)")
            except Exception as e:
                print(f"   ⚠️  ML preprocessor failed: {e}")
                self.use_ml = False
                print("   ℹ️  Falling back to hard-coded corrections")
        else:
            print("\n4️⃣  ML Text Corrector: Disabled")
            print("   ℹ️  Using hard-coded corrections only")
        
        print("\n" + "="*70)
        print("✅ PIPELINE READY")
        print("="*70 + "\n")
    
    def detect_input_type(self, file_path: str) -> str:
        """Detect input file type"""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        type_map = {
            '.txt': 'text',
            '.docx': 'docx',
            '.pdf': 'pdf',
            '.png': 'image',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.tiff': 'image',
            '.bmp': 'image'
        }
        
        return type_map.get(ext, 'unknown')
    
    def process_document(
        self,
        input_path: str,
        document_name: Optional[str] = None
    ) -> Dict:
        """
        Process a single document through the complete pipeline
        
        Args:
            input_path: Path to input file
            document_name: Optional custom name
            
        Returns:
            Processing results dictionary
        """
        start_time = time.time()
        
        input_path = Path(input_path)
        if document_name is None:
            document_name = input_path.stem
        
        print("\n" + "="*70)
        print(f"PROCESSING: {document_name}")
        print("="*70)
        
        result = {
            'document_name': document_name,
            'input_path': str(input_path),
            'input_type': None,
            'extraction_method': None,
            'success': False,
            'stages': {},
            'processing_time': 0
        }
        
        try:
            # Stage 1: Detect input type
            print("\n📋 STAGE 1: Input Detection")
            input_type = self.detect_input_type(str(input_path))
            result['input_type'] = input_type
            print(f"   Type: {input_type.upper()}")
            result['stages']['detection'] = {'success': True, 'type': input_type}
            
            # Stage 2: Text Extraction
            print("\n📄 STAGE 2: Text Extraction")
            if input_type == 'text':
                # Direct text file
                raw_text = input_path.read_text(encoding='utf-8')
                result['extraction_method'] = 'direct_text'
                print(f"   Method: Direct text read")
                print(f"   ✅ Extracted: {len(raw_text.split())} words")
            
            elif input_type == 'docx':
                # DOCX extraction
                raw_text = self.extractor.extract_text_from_docx(input_path)
                result['extraction_method'] = 'docx_direct'
                print(f"   Method: Direct DOCX extraction")
                print(f"   ✅ Extracted: {len(raw_text.split())} words")
            
            elif input_type == 'pdf':
                # PDF - check if needs OCR
                print("   Analyzing PDF quality...")
                # Try direct extraction first
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(str(input_path))
                    sample_text = ""
                    for page in reader.pages[:3]:  # Check first 3 pages
                        sample_text += page.extract_text()
                    
                    word_count = len(sample_text.split())
                    
                    if word_count > 50:
                        # Good quality, use direct extraction
                        raw_text = sample_text
                        result['extraction_method'] = 'pdf_direct'
                        print(f"   Method: Direct PDF extraction (good quality)")
                        print(f"   ✅ Extracted: {word_count} words")
                    else:
                        # Low quality, needs OCR
                        print(f"   Method: OCR path (scanned PDF)")
                        raw_text = self._process_with_ocr(input_path, 'pdf')
                        result['extraction_method'] = 'pdf_ocr'
                        print(f"   ✅ OCR completed: {len(raw_text.split())} words")
                
                except Exception as e:
                    print(f"   ⚠️  Direct extraction failed, using OCR")
                    raw_text = self._process_with_ocr(input_path, 'pdf')
                    result['extraction_method'] = 'pdf_ocr'
            
            elif input_type == 'image':
                # Image - use Tesseract OCR for printed text
                print(f"   Method: Tesseract OCR")
                raw_text = self._process_with_ocr(input_path, 'image')
                result['extraction_method'] = 'image_ocr'
                print(f"   ✅ OCR completed: {len(raw_text.split())} words")
            
            else:
                raise ValueError(f"Unsupported input type: {input_type}")
            
            result['stages']['extraction'] = {
                'success': True,
                'method': result['extraction_method'],
                'word_count': len(raw_text.split())
            }
            
            # Stage 3: Text Correction & Preprocessing
            print("\n🔧 STAGE 3: Text Correction")
            
            if self.use_ml and self.ml_profile == 'symspell':
                # ML-based correction
                print("   Method: ML-based (SymSpell)")
                corrected_text = self.ml_preprocessor.preprocess(raw_text)
                print(f"   ✅ ML corrected: {len(corrected_text.split())} words")
            
            elif self.ml_profile == 'both':
                # Both corrections
                print("   Method: Hard-coded + ML")
                temp_text = self.text_preprocessor.preprocess(raw_text)
                if self.use_ml:
                    corrected_text = self.ml_preprocessor.preprocess(temp_text)
                else:
                    corrected_text = temp_text
                print(f"   ✅ Dual corrected: {len(corrected_text.split())} words")
            
            else:
                # Hard-coded only
                print("   Method: Hard-coded corrections")
                corrected_text = self.text_preprocessor.preprocess(raw_text)
                print(f"   ✅ Corrected: {len(corrected_text.split())} words")
            
            result['stages']['correction'] = {
                'success': True,
                'method': self.ml_profile if self.use_ml else 'hard-coded',
                'word_count': len(corrected_text.split())
            }
            
            # Stage 4: Save Output
            print("\n💾 STAGE 4: Save Output")
            output_file = self.final_output / f"{document_name}.txt"
            output_file.write_text(corrected_text, encoding='utf-8')
            print(f"   ✅ Saved: {output_file}")
            
            result['stages']['output'] = {
                'success': True,
                'file': str(output_file)
            }
            
            # Final result
            result['success'] = True
            result['output_file'] = str(output_file)
            result['text'] = corrected_text
            result['word_count'] = len(corrected_text.split())
            result['processing_time'] = time.time() - start_time
            
            print("\n" + "="*70)
            print(f"✅ SUCCESS - Processed in {result['processing_time']:.2f} seconds")
            print("="*70)
        
        except Exception as e:
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
            print(f"\n❌ ERROR: {e}")
            print("="*70)
        
        return result
    
    def _process_with_ocr(self, file_path: Path, file_type: str) -> str:
        """Process file with OCR pipeline"""
        
        if file_type == 'image':
            # Image → Preprocess → OCR
            print("   ├─ Image preprocessing...")
            preprocessed_cv, _ = self.image_preprocessor.preprocess(
                str(file_path),
                profile='standard'
            )
            
            print("   ├─ Running OCR...")
            import pytesseract
            from PIL import Image
            import cv2
            
            # Convert CV to PIL
            rgb = cv2.cvtColor(preprocessed_cv, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            
            # Use better Tesseract config for structured documents
            # PSM 6 = Assume uniform block of text (good for reports/forms)
            text = pytesseract.image_to_string(pil_image, lang='eng', config='--psm 6')
            
        elif file_type == 'pdf':
            # PDF → Images → Preprocess → OCR
            print("   ├─ Converting PDF to images...")
            from pdf2image import convert_from_path
            poppler_path = r"C:\poppler\poppler-24.08.0\Library\bin"
            images = convert_from_path(str(file_path), dpi=300, poppler_path=poppler_path)
            
            all_text = []
            for i, image in enumerate(images, 1):
                print(f"   ├─ Processing page {i}/{len(images)}...")
                
                # Preprocess
                preprocessed = self.image_preprocessor.preprocess(image, profile='standard')
                
                # OCR with better config for structured documents
                import pytesseract
                page_text = pytesseract.image_to_string(preprocessed, lang='eng', config='--psm 6')
                all_text.append(page_text)
            
            text = "\n\n".join(all_text)
        
        return text
    
    def process_batch(
        self,
        input_dir: str,
        pattern: str = "*.*",
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Process all files in a directory
        
        Args:
            input_dir: Input directory
            pattern: File pattern (e.g., "*.docx")
            limit: Max files to process
            
        Returns:
            List of processing results
        """
        input_path = Path(input_dir)
        files = sorted(input_path.glob(pattern))
        
        if limit:
            files = files[:limit]
        
        print("\n" + "="*70)
        print(f"BATCH PROCESSING: {len(files)} files")
        print("="*70)
        
        results = []
        
        for i, file in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}]")
            result = self.process_document(str(file))
            results.append(result)
        
        # Summary
        self._print_batch_summary(results)
        
        return results
    
    def _print_batch_summary(self, results: List[Dict]):
        """Print batch processing summary"""
        print("\n" + "="*70)
        print("BATCH PROCESSING SUMMARY")
        print("="*70)
        
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        total_words = sum(r.get('word_count', 0) for r in results if r['success'])
        total_time = sum(r.get('processing_time', 0) for r in results)
        
        print(f"\n📊 Results:")
        print(f"   Total Files: {len(results)}")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📝 Total Words: {total_words:,}")
        print(f"   ⏱️  Total Time: {total_time:.2f} seconds")
        print(f"   ⚡ Avg Speed: {total_time/len(results):.2f} sec/file")
        
        # Method breakdown
        methods = {}
        for r in results:
            if r['success']:
                method = r.get('extraction_method', 'unknown')
                methods[method] = methods.get(method, 0) + 1
        
        if methods:
            print(f"\n📈 Extraction Methods:")
            for method, count in methods.items():
                print(f"   {method}: {count} files")
        
        print("\n📂 Output Directory:")
        print(f"   {self.final_output}")
        print("="*70 + "\n")


# =================================================================
# CONVENIENCE FUNCTIONS
# =================================================================

def quick_process(
    input_path: str,
    use_ml: bool = True
) -> Dict:
    """
    Quick function to process a single document
    
    Args:
        input_path: Path to input file
        use_ml: Use ML-based correction
        
    Returns:
        Processing result
    """
    pipeline = CompletePipeline(use_ml_correction=use_ml)
    return pipeline.process_document(input_path)


def batch_process(
    input_dir: str = "archive",
    pattern: str = "*.docx",
    use_ml: bool = True,
    limit: Optional[int] = None
) -> List[Dict]:
    """
    Quick function to process multiple documents
    
    Args:
        input_dir: Input directory
        pattern: File pattern
        use_ml: Use ML-based correction
        limit: Max files to process
        
    Returns:
        List of processing results
    """
    pipeline = CompletePipeline(use_ml_correction=use_ml)
    return pipeline.process_batch(input_dir, pattern, limit)


# =================================================================
# MAIN - DEMO & TESTING
# =================================================================

if __name__ == '__main__':
    import sys
    
    print("\n" + "="*70)
    print("COMPLETE OCR PIPELINE - PRODUCTION READY")
    print("="*70)
    
    print("\n🎯 What would you like to do?")
    print("\n1. Process a single file")
    print("2. Process all files in archive/ directory")
    print("3. Process all files (with ML correction)")
    print("4. Run demo with test files")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        file_path = input("Enter file path: ").strip()
        result = quick_process(file_path, use_ml=ML_AVAILABLE)
        
        if result['success']:
            print(f"\n✅ Successfully processed!")
            print(f"   Output: {result['output_file']}")
            print(f"   Words: {result['word_count']}")
    
    elif choice == '2':
        results = batch_process(
            input_dir='archive',
            pattern='*.docx',
            use_ml=False  # Fast processing
        )
    
    elif choice == '3':
        if not ML_AVAILABLE:
            print("\n⚠️  ML correction not available. Install symspellpy:")
            print("   pip install symspellpy")
            sys.exit(1)
        
        results = batch_process(
            input_dir='archive',
            pattern='*.docx',
            use_ml=True  # With ML correction
        )
    
    elif choice == '4':
        # Demo mode
        print("\n🎬 DEMO MODE")
        print("\nThis will process sample files from output/final_preprocessed/")
        
        pipeline = CompletePipeline(use_ml_correction=ML_AVAILABLE)
        
        # Find test files
        test_dir = Path('output/final_preprocessed')
        if test_dir.exists():
            test_files = list(test_dir.glob('*.txt'))[:3]
            
            if test_files:
                for file in test_files:
                    result = pipeline.process_document(str(file))
            else:
                print("\n⚠️  No test files found")
        else:
            print("\n⚠️  Test directory not found")
    
    else:
        print("Invalid choice!")
