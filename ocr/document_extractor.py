"""
Module 1: Complete Document Extraction System
==============================================

Dual Path Implementation:
1. Direct .docx text extraction (fast, clean)
2. OCR path: .docx → PDF → Images → OCR (handles scanned documents)

Features:
- Batch processing with progress tracking
- Automatic format detection
- Text cleaning and preprocessing
- Comprehensive error handling
- Detailed reporting

Author: AI Assistant
Date: October 8, 2025
"""

import os
import re
from pathlib import Path
from docx import Document
from PIL import Image

# Optional imports for OCR path
try:
    import win32com.client
    import pythoncom
    WORD_COM_AVAILABLE = True
except ImportError:
    WORD_COM_AVAILABLE = False
    print("⚠️  Warning: pywin32 not installed. OCR path disabled.")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("⚠️  Warning: pdf2image not installed. OCR path disabled.")

try:
    import pytesseract
    # Configure Tesseract path (use environment variable or default Windows path)
    pytesseract.pytesseract.tesseract_cmd = os.environ.get(
        'TESSERACT_CMD', 
        r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    )
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("⚠️  Warning: pytesseract not installed. OCR path disabled.")


class CompleteDocumentExtractor:
    """
    Complete document extraction system with dual paths:
    - Direct extraction from .docx
    - OCR extraction via PDF and images
    """
    
    def __init__(self, archive_dir="archive", output_base_dir="."):
        """
        Initialize the complete extractor
        
        Args:
            archive_dir: Directory containing .docx files
            output_base_dir: Base directory for all outputs
        """
        self.archive_dir = Path(archive_dir)
        self.output_base_dir = Path(output_base_dir)
        
        # Create output directories
        self.text_extracted_dir = self.output_base_dir / "text_extracted"
        self.text_extracted_ocr_dir = self.output_base_dir / "text_extracted_ocr"
        self.images_dir = self.output_base_dir / "images"
        self.pdf_temp_dir = self.output_base_dir / "pdf_temp"
        
        for directory in [self.text_extracted_dir, self.text_extracted_ocr_dir, 
                          self.images_dir, self.pdf_temp_dir]:
            directory.mkdir(exist_ok=True)
        
        # Check OCR availability
        self.ocr_available = WORD_COM_AVAILABLE and PDF2IMAGE_AVAILABLE and PYTESSERACT_AVAILABLE
    
    # =================================================================
    # DIRECT EXTRACTION PATH
    # =================================================================
    
    def extract_text_from_docx(self, docx_path):
        """
        Direct extraction from .docx file
        
        Args:
            docx_path: Path to .docx file
            
        Returns:
            Cleaned text string
        """
        doc = Document(docx_path)
        
        # Extract all paragraphs
        paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:  # Skip empty paragraphs
                paragraphs.append(text)
        
        # Combine into single text block
        full_text = "\n".join(paragraphs)
        
        # Clean the text
        cleaned_text = self._clean_text(full_text)
        
        return cleaned_text
    
    def _clean_text(self, text):
        """
        Remove headers, footers, page numbers, and extra whitespace
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        # Remove common page number patterns
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Page numbers on separate lines
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple blank lines → double newline
        text = re.sub(r' +', ' ', text)  # Multiple spaces → single space
        
        # Remove common header/footer artifacts
        text = re.sub(r'^[-_=]+$', '', text, flags=re.MULTILINE)  # Lines of dashes/underscores
        
        return text.strip()
    
    # =================================================================
    # OCR EXTRACTION PATH
    # =================================================================
    
    def convert_docx_to_pdf(self, docx_path, pdf_path):
        """
        Convert .docx to PDF using Microsoft Word COM
        
        Args:
            docx_path: Path to .docx file
            pdf_path: Path to output PDF
            
        Returns:
            True if successful, False otherwise
        """
        if not WORD_COM_AVAILABLE:
            return False
            
        try:
            # Initialize COM
            pythoncom.CoInitialize()
            
            # Create Word application
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            
            # Open document
            doc = word.Documents.Open(str(docx_path.absolute()))
            
            # Save as PDF (format 17 = wdFormatPDF)
            doc.SaveAs(str(pdf_path.absolute()), FileFormat=17)
            
            # Close document and Word
            doc.Close()
            word.Quit()
            
            # Cleanup COM
            pythoncom.CoUninitialize()
            
            return True
            
        except Exception as e:
            try:
                word.Quit()
                pythoncom.CoUninitialize()
            except:
                pass
            return False
    
    def convert_pdf_to_images(self, pdf_path, doc_name):
        """
        Convert PDF pages to PNG images
        
        Args:
            pdf_path: Path to PDF file
            doc_name: Base name for output images
            
        Returns:
            List of image paths
        """
        if not PDF2IMAGE_AVAILABLE:
            return []
            
        try:
            # Poppler path (use environment variable or default Windows path)
            poppler_path = os.environ.get('POPPLER_PATH', r"C:\poppler\poppler-24.08.0\Library\bin")
            
            # Convert PDF to images (300 DPI for good quality)
            images = convert_from_path(str(pdf_path), dpi=300, poppler_path=poppler_path)
            
            image_paths = []
            for i, image in enumerate(images, start=1):
                image_path = self.images_dir / f"{doc_name}_page_{i:03d}.png"
                image.save(str(image_path), "PNG")
                image_paths.append(image_path)
            
            return image_paths
            
        except Exception as e:
            print(f"⚠️  PDF to image conversion failed: {e}")
            return []
    
    def extract_text_from_image_ocr(self, image_path):
        """
        Extract text from image using Tesseract OCR
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text string
        """
        if not PYTESSERACT_AVAILABLE:
            raise Exception("pytesseract not available")
            
        try:
            # Open image
            image = Image.open(image_path)
            
            # Run OCR
            text = pytesseract.image_to_string(image, lang='eng')
            
            return text
            
        except Exception as e:
            raise Exception(f"OCR failed: {e}")
    
    def _clean_ocr_text(self, text):
        """
        Clean OCR-specific artifacts and noise
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[|]{2,}', '', text)  # Multiple pipes
        text = re.sub(r'_{3,}', '', text)    # Multiple underscores
        text = re.sub(r'={3,}', '', text)    # Multiple equals
        
        # Remove page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        
        # Remove single character lines (often OCR noise)
        text = re.sub(r'\n\s*[^\w\s]\s*\n', '\n', text)
        
        return text.strip()
    
    # =================================================================
    # PROCESSING METHODS
    # =================================================================
    
    def process_single_document(self, docx_path, enable_ocr=True):
        """
        Process a single document through both paths
        
        Args:
            docx_path: Path to .docx file
            enable_ocr: Whether to run OCR path (default: True)
            
        Returns:
            Dictionary with results from both paths
        """
        doc_name = docx_path.stem
        print(f"\n📄 Processing: {doc_name}")
        
        results = {
            "document": doc_name,
            "direct_extraction": None,
            "ocr_extraction": None,
            "pdf_created": False,
            "images_created": 0
        }
        
        # ============ PATH 1: Direct .docx extraction ============
        try:
            print("   ├─ Direct extraction...", end=" ", flush=True)
            text = self.extract_text_from_docx(docx_path)
            output_path = self.text_extracted_dir / f"{doc_name}.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            results["direct_extraction"] = {
                "success": True,
                "output_path": str(output_path),
                "word_count": len(text.split())
            }
            print(f"✓ ({len(text.split())} words)")
        except Exception as e:
            print(f"✗ Error: {e}")
            results["direct_extraction"] = {"success": False, "error": str(e)}
        
        # ============ PATH 2: OCR extraction ============
        if enable_ocr and self.ocr_available:
            try:
                # Step 1: Convert .docx to PDF
                print("   ├─ Converting to PDF...", end=" ", flush=True)
                pdf_path = self.pdf_temp_dir / f"{doc_name}.pdf"
                if self.convert_docx_to_pdf(docx_path, pdf_path):
                    results["pdf_created"] = True
                    print("✓")
                else:
                    print("✗")
                    results["ocr_extraction"] = {"success": False, "error": "PDF conversion failed"}
                    print("   └─ Done!")
                    return results
                
                # Step 2: Convert PDF to images
                print("   ├─ Converting to images...", end=" ", flush=True)
                image_paths = self.convert_pdf_to_images(pdf_path, doc_name)
                results["images_created"] = len(image_paths)
                if image_paths:
                    print(f"✓ ({len(image_paths)} pages)")
                else:
                    print("✗")
                    results["ocr_extraction"] = {"success": False, "error": "Image conversion failed"}
                    print("   └─ Done!")
                    return results
                
                # Step 3: OCR each image
                print("   ├─ Running OCR...", end=" ", flush=True)
                all_text = []
                ocr_errors = []
                for image_path in image_paths:
                    try:
                        text = self.extract_text_from_image_ocr(image_path)
                        if text:
                            all_text.append(text)
                    except Exception as e:
                        ocr_errors.append(str(e))
                
                if not all_text:
                    error_msg = f"No text extracted. Errors: {'; '.join(ocr_errors)}" if ocr_errors else "No text extracted"
                    print(f"✗ ({error_msg})")
                    results["ocr_extraction"] = {"success": False, "error": error_msg}
                    print("   └─ Done!")
                    return results
                
                # Combine and clean
                combined_text = "\n\n".join(all_text)
                cleaned_text = self._clean_ocr_text(combined_text)
                
                # Save to file
                output_path = self.text_extracted_ocr_dir / f"{doc_name}.txt"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                
                results["ocr_extraction"] = {
                    "success": True,
                    "output_path": str(output_path),
                    "word_count": len(cleaned_text.split())
                }
                print(f"✓ ({results['ocr_extraction']['word_count']} words)")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                results["ocr_extraction"] = {"success": False, "error": str(e)}
        elif enable_ocr and not self.ocr_available:
            print("   ├─ OCR path skipped (dependencies not installed)")
        
        print("   └─ Done!")
        return results
    
    def process_all_documents(self, enable_ocr=True, limit=None):
        """
        Process all documents in the archive directory
        
        Args:
            enable_ocr: Whether to run OCR path (default: True)
            limit: Maximum number of documents to process (None = all)
            
        Returns:
            List of results for all documents
        """
        # Get all .docx files
        docx_files = sorted(self.archive_dir.glob("*.docx"))
        
        if limit:
            docx_files = docx_files[:limit]
        
        print(f"\n{'='*70}")
        print(f"COMPLETE DOCUMENT EXTRACTION SYSTEM")
        print(f"{'='*70}")
        print(f"📁 Archive: {self.archive_dir}")
        print(f"📄 Documents: {len(docx_files)}")
        print(f"🔧 Direct Path: Enabled")
        print(f"🔧 OCR Path: {'Enabled' if (enable_ocr and self.ocr_available) else 'Disabled'}")
        if enable_ocr and not self.ocr_available:
            print(f"   ⚠️  OCR dependencies not fully installed")
        print(f"{'='*70}")
        
        all_results = []
        
        for i, docx_path in enumerate(docx_files, 1):
            print(f"\n[{i}/{len(docx_files)}]", end=" ")
            result = self.process_single_document(docx_path, enable_ocr=enable_ocr)
            all_results.append(result)
        
        # Print summary
        self._print_summary(all_results, enable_ocr)
        
        return all_results
    
    def _print_summary(self, results, ocr_enabled):
        """
        Print processing summary
        
        Args:
            results: List of result dictionaries
            ocr_enabled: Whether OCR was enabled
        """
        print(f"\n{'='*70}")
        print("PROCESSING SUMMARY")
        print(f"{'='*70}")
        
        # Direct extraction stats
        direct_success = sum(1 for r in results if r["direct_extraction"] and r["direct_extraction"]["success"])
        direct_words = sum(r["direct_extraction"]["word_count"] 
                          for r in results 
                          if r["direct_extraction"] and r["direct_extraction"]["success"])
        
        print(f"✅ Direct Extraction: {direct_success}/{len(results)} successful")
        print(f"   📝 Total Words: {direct_words:,}")
        
        # OCR extraction stats (if enabled)
        if ocr_enabled and self.ocr_available:
            pdf_success = sum(1 for r in results if r["pdf_created"])
            total_images = sum(r["images_created"] for r in results)
            ocr_success = sum(1 for r in results if r["ocr_extraction"] and r["ocr_extraction"]["success"])
            ocr_words = sum(r["ocr_extraction"]["word_count"] 
                           for r in results 
                           if r["ocr_extraction"] and r["ocr_extraction"]["success"])
            
            print(f"\n✅ OCR Extraction: {ocr_success}/{len(results)} successful")
            print(f"   📄 PDFs Created: {pdf_success}")
            print(f"   🖼️  Images Created: {total_images}")
            print(f"   📝 Total Words: {ocr_words:,}")
        
        print(f"\n📂 Output Directories:")
        print(f"   • text_extracted/     → {direct_success} files")
        if ocr_enabled and self.ocr_available:
            print(f"   • text_extracted_ocr/ → {ocr_success} files")
            print(f"   • pdf_temp/           → {pdf_success} files")
            print(f"   • images/             → {total_images} images")
        print(f"{'='*70}\n")


# =================================================================
# CONVENIENCE FUNCTIONS
# =================================================================

def extract_direct_only(archive_dir="archive", limit=None):
    """
    Quick function: Extract text directly from .docx files only
    
    Args:
        archive_dir: Directory containing .docx files
        limit: Max documents to process (None = all)
        
    Returns:
        List of results
    """
    extractor = CompleteDocumentExtractor(archive_dir=archive_dir)
    return extractor.process_all_documents(enable_ocr=False, limit=limit)


def extract_with_ocr(archive_dir="archive", limit=None):
    """
    Quick function: Extract text using both direct and OCR paths
    Both paths run automatically - saves to separate folders:
    - text_extracted/ (direct extraction)
    - text_extracted_ocr/ (OCR extraction)
    
    Args:
        archive_dir: Directory containing .docx files
        limit: Max documents to process (None = all)
        
    Returns:
        List of results
    """
    extractor = CompleteDocumentExtractor(archive_dir=archive_dir)
    return extractor.process_all_documents(enable_ocr=True, limit=limit)


def extract_both_paths(archive_dir="archive", limit=None):
    """
    Recommended: Extract using BOTH direct and OCR paths
    Creates two separate output folders for comparison:
    - text_extracted/ (clean, fast direct extraction)
    - text_extracted_ocr/ (OCR from images)
    
    Args:
        archive_dir: Directory containing .docx files
        limit: Max documents to process (None = all)
        
    Returns:
        List of results with both extraction methods
    """
    print("\n🎯 Running DUAL PATH extraction:")
    print("   📝 Path 1: Direct text extraction (.docx → text)")
    print("   🖼️  Path 2: OCR extraction (.docx → PDF → Images → OCR text)")
    print("")
    
    extractor = CompleteDocumentExtractor(archive_dir=archive_dir)
    return extractor.process_all_documents(enable_ocr=True, limit=limit)


def main():
    """
    Main function - Interactive menu
    """
    print("\n" + "="*70)
    print("COMPLETE DOCUMENT EXTRACTION SYSTEM")
    print("="*70)
    print("\nAvailable Options:")
    print("  1. BOTH paths (direct + OCR) - Recommended for complete demo")
    print("  2. Direct extraction only (fast, for production)")
    print("  3. OCR extraction only (for scanned documents)")
    print("\nDependency Status:")
    print(f"  • Direct extraction: ✅ Ready")
    print(f"  • Word COM: {'✅' if WORD_COM_AVAILABLE else '❌'} {'Ready' if WORD_COM_AVAILABLE else 'Not installed (pip install pywin32)'}")
    print(f"  • pdf2image: {'✅' if PDF2IMAGE_AVAILABLE else '❌'} {'Ready' if PDF2IMAGE_AVAILABLE else 'Not installed (pip install pdf2image)'}")
    print(f"  • Tesseract OCR: {'✅' if PYTESSERACT_AVAILABLE else '❌'} {'Ready' if PYTESSERACT_AVAILABLE else 'Not installed (pip install pytesseract)'}")
    
    print("\n" + "="*70)
    print("Quick Start Examples:")
    print("="*70)
    print("\n# RECOMMENDED: Run both extraction paths (creates 2 output folders):")
    print("from module1_complete import extract_both_paths")
    print("results = extract_both_paths()  # Saves to text_extracted/ AND text_extracted_ocr/")
    print("\n# Fast direct extraction only:")
    print("from module1_complete import extract_direct_only")
    print("results = extract_direct_only()  # Saves to text_extracted/ only")
    print("\n# Custom processing:")
    print("from module1_complete import CompleteDocumentExtractor")
    print("extractor = CompleteDocumentExtractor()")
    print("results = extractor.process_all_documents(enable_ocr=True)  # Both paths")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
