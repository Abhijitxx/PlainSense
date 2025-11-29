"""
Enhanced Clause Segmentation Module v2
======================================

Major Improvements:
1. Better sentence boundary detection (legal-aware)
2. Hierarchical clustering with optimal threshold
3. Improved clause type classification
4. Context-aware merging
5. Support for numbered/bulleted clauses

Author: PlainSense Team
Date: November 2025
"""

import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ClauseType(Enum):
    """Types of clauses in rental agreements"""
    # Primary clause types
    HEADER = "header"
    PARTIES = "parties"
    PROPERTY = "property"
    RENT = "rent"
    DEPOSIT = "deposit"
    DURATION = "duration"
    TERMINATION = "termination"
    MAINTENANCE = "maintenance"
    UTILITIES = "utilities"
    RESTRICTIONS = "restrictions"
    SUBLETTING = "subletting"
    PENALTIES = "penalties"
    OBLIGATIONS = "obligations"
    DISPUTE = "dispute"
    SIGNATURES = "signatures"
    GENERAL = "general"
    UNKNOWN = "unknown"


@dataclass
class Clause:
    """Represents a document clause"""
    id: int
    clause_type: ClauseType
    title: str
    content: str
    sentences: List[str]
    start_idx: int
    end_idx: int
    confidence: float
    risk_score: float = 0.0
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['clause_type'] = self.clause_type.value
        return result


class EnhancedClauseSegmenter:
    """
    Enhanced clause segmentation with hierarchical clustering
    """
    
    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.60,
                 use_gpu: bool = True):
        """
        Initialize segmenter
        
        Args:
            embedding_model: Sentence transformer model
            similarity_threshold: Threshold for merging sentences
            use_gpu: Use GPU if available
        """
        self.similarity_threshold = similarity_threshold
        self.device = "cuda" if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        
        # Load embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print(f"🔄 Loading clause segmenter: {embedding_model}...")
            self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
            print(f"   ✅ Loaded on {self.device}")
        else:
            self.embedding_model = None
            print("   ⚠️ Sentence transformers not available")
        
        # Initialize clause type prototypes
        self.clause_prototypes = {}
        self.clause_embeddings = {}
        self._init_clause_prototypes()
        
        # Legal sentence patterns
        self.legal_patterns = self._init_legal_patterns()
    
    def _init_legal_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for legal documents"""
        return {
            'numbered': re.compile(r'^(\d+[\.\)]\s*|\([a-z]\)\s*|[a-z]\)\s*|•\s*|-\s*)(.+)', re.IGNORECASE),
            'clause_header': re.compile(r'^(CLAUSE\s+\d+|Article\s+\d+|Section\s+\d+)[:\s]*(.+)?', re.IGNORECASE),
            'whereas': re.compile(r'^WHEREAS\s+', re.IGNORECASE),
            'agreement': re.compile(r'(agreement|contract|lease|rental)\s+(is\s+)?(made|entered|executed)', re.IGNORECASE),
            'witness': re.compile(r'(WITNESS|IN\s+WITNESS\s+WHEREOF)', re.IGNORECASE),
            'signature': re.compile(r'(signature|signed|witnessed)', re.IGNORECASE),
        }
    
    def _init_clause_prototypes(self):
        """Initialize prototype embeddings for clause classification"""
        
        # Representative examples for each clause type
        self.clause_examples = {
            ClauseType.HEADER: [
                "This rental agreement is made on this day",
                "Agreement for lease of residential premises",
                "This lease deed is executed between",
                "Rental agreement dated",
            ],
            ClauseType.PARTIES: [
                "Between the landlord hereinafter called lessor",
                "And the tenant hereinafter called lessee",
                "First party owner of the property",
                "Second party agrees to take on rent",
                "Mr Ramesh Kumar residing at Bangalore",
            ],
            ClauseType.PROPERTY: [
                "The premises described in the schedule",
                "Flat number 302 located at Green Park",
                "All that piece and parcel of property",
                "The demised premises situated at",
                "Property address and description",
            ],
            ClauseType.RENT: [
                "Monthly rent of rupees fifteen thousand",
                "Rent payable on or before fifth of every month",
                "The tenant shall pay monthly rent",
                "Payment of rent in advance",
                "Rent amount Rs 25000 per month",
            ],
            ClauseType.DEPOSIT: [
                "Security deposit of one lakh rupees",
                "Advance amount paid as caution money",
                "Deposit refundable at end of tenancy",
                "Interest free security deposit",
                "Deposit to be returned within one month",
            ],
            ClauseType.DURATION: [
                "Period of lease shall be eleven months",
                "Agreement valid from January to December",
                "Tenancy period of one year",
                "Lease commences from first of the month",
                "Lock in period of six months",
            ],
            ClauseType.TERMINATION: [
                "Either party may terminate with notice",
                "Three months notice required for termination",
                "Agreement can be terminated by written notice",
                "Tenant shall surrender vacant possession",
                "Lease terminates on expiry of period",
            ],
            ClauseType.MAINTENANCE: [
                "Tenant shall maintain premises in good condition",
                "Repairs and maintenance responsibility",
                "Landlord responsible for structural repairs",
                "Regular maintenance by tenant",
                "Keep the property in tenantable condition",
            ],
            ClauseType.UTILITIES: [
                "Electricity charges to be paid by tenant",
                "Water charges at actuals",
                "Utility bills responsibility of lessee",
                "Maintenance charges paid to society",
                "EB and water meter reading",
            ],
            ClauseType.RESTRICTIONS: [
                "Tenant shall not make alterations",
                "No modifications without written consent",
                "Cannot use for commercial purposes",
                "Residential use only",
                "No illegal activities on premises",
            ],
            ClauseType.SUBLETTING: [
                "Tenant shall not sublet the premises",
                "No subletting without permission",
                "Cannot part with possession",
                "Not let out to any third party",
                "Sublease not permitted",
            ],
            ClauseType.PENALTIES: [
                "Penalty for late payment of rent",
                "Damages for breach of agreement",
                "Fine for early termination",
                "Compensation for violation",
                "Liable for all costs and expenses",
            ],
            ClauseType.OBLIGATIONS: [
                "Tenant agrees to the following",
                "Landlord shall ensure peaceful possession",
                "Duties and responsibilities of parties",
                "Both parties mutually agree",
                "Covenants and conditions",
            ],
            ClauseType.SIGNATURES: [
                "In witness whereof parties have signed",
                "Signed sealed and delivered",
                "Witnessed by the following persons",
                "Signature of landlord tenant",
                "Date and place of execution",
            ],
        }
        
        # Compute embeddings
        if self.embedding_model:
            print("   Computing clause type prototypes...")
            for clause_type, examples in self.clause_examples.items():
                embeddings = self.embedding_model.encode(examples)
                self.clause_embeddings[clause_type] = np.mean(embeddings, axis=0)
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences with legal document awareness
        """
        # Preserve numbered items
        text = re.sub(r'(\d+)\.\s+', r'\1) ', text)
        
        # Handle common abbreviations to prevent false splits
        abbrevs = ['mr', 'mrs', 'ms', 'dr', 'sr', 'jr', 'no', 'nos', 'rs', 'vs', 'etc', 'ie', 'eg']
        for abbr in abbrevs:
            text = re.sub(rf'\b{abbr}\.', f'{abbr}ABBREV', text, flags=re.IGNORECASE)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=\d+[\)\.]\s)|(?<=\n)\s*(?=[A-Z])', text)
        
        # Restore abbreviations
        sentences = [s.replace('ABBREV', '.') for s in sentences]
        
        # Clean and filter
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            sent = re.sub(r'\s+', ' ', sent)
            if len(sent) > 10:  # Minimum sentence length
                cleaned.append(sent)
        
        return cleaned
    
    def _detect_numbered_clauses(self, text: str) -> List[Tuple[int, str, str]]:
        """
        Detect numbered or bulleted clauses
        
        Returns:
            List of (number, content, original_text) tuples
        """
        numbered_clauses = []
        
        # Pattern for numbered items
        pattern = re.compile(
            r'^(\d+)[\.\)]\s*(.+?)(?=^\d+[\.\)]|\Z)',
            re.MULTILINE | re.DOTALL
        )
        
        for match in pattern.finditer(text):
            number = int(match.group(1))
            content = match.group(2).strip()
            numbered_clauses.append((number, content, match.group(0)))
        
        return numbered_clauses
    
    def _classify_clause_type(self, text: str) -> Tuple[ClauseType, float]:
        """
        Classify clause type using embedding similarity
        
        Returns:
            (ClauseType, confidence)
        """
        if not self.clause_embeddings:
            return ClauseType.GENERAL, 0.5
        
        # Get embedding for clause text
        text_embedding = self.embedding_model.encode(text[:500])
        
        # Compute similarities
        similarities = {}
        for clause_type, prototype in self.clause_embeddings.items():
            sim = float(cosine_similarity(
                [text_embedding], [prototype]
            )[0][0])
            similarities[clause_type] = sim
        
        # Get best match
        best_type = max(similarities, key=similarities.get)
        confidence = similarities[best_type]
        
        # Use keyword boost for certain types
        text_lower = text.lower()
        keyword_boosts = {
            ClauseType.RENT: ['rent', 'rupees', 'rs.', 'payment', 'monthly'],
            ClauseType.DEPOSIT: ['deposit', 'advance', 'security', 'caution'],
            ClauseType.TERMINATION: ['terminate', 'termination', 'notice', 'vacate'],
            ClauseType.SUBLETTING: ['sublet', 'sub-let', 'sublease'],
            ClauseType.PENALTIES: ['penalty', 'fine', 'damages', 'compensation'],
            ClauseType.SIGNATURES: ['witness', 'signed', 'signature'],
        }
        
        for ctype, keywords in keyword_boosts.items():
            if any(kw in text_lower for kw in keywords):
                similarities[ctype] = similarities.get(ctype, 0) + 0.1
        
        # Re-check best match after boost
        best_type = max(similarities, key=similarities.get)
        confidence = min(similarities[best_type], 1.0)
        
        if confidence < 0.4:
            return ClauseType.GENERAL, confidence
        
        return best_type, confidence
    
    def _cluster_sentences(self, 
                           sentences: List[str],
                           embeddings: np.ndarray) -> List[List[int]]:
        """
        Cluster sentences into clauses using hierarchical clustering
        
        Returns:
            List of sentence index groups
        """
        if len(sentences) <= 1:
            return [[0]] if sentences else []
        
        if len(sentences) <= 3:
            return [[i] for i in range(len(sentences))]
        
        # Compute pairwise similarities
        similarity_matrix = cosine_similarity(embeddings)
        
        # Convert to distance matrix
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)
        
        # Hierarchical clustering
        try:
            # Use Ward's method for compact clusters
            linkage_matrix = linkage(distance_matrix, method='average')
            
            # Dynamic threshold based on document length
            threshold = self.similarity_threshold
            if len(sentences) > 20:
                threshold = min(threshold + 0.05, 0.75)
            
            # Cut tree at threshold
            cluster_labels = fcluster(
                linkage_matrix, 
                t=1 - threshold,  # Convert similarity to distance
                criterion='distance'
            )
            
            # Group sentences by cluster
            clusters = {}
            for idx, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)
            
            # Sort clusters by first sentence index (maintain order)
            sorted_clusters = sorted(clusters.values(), key=lambda x: min(x))
            
            return sorted_clusters
            
        except Exception as e:
            print(f"   ⚠️ Clustering error: {e}")
            # Fallback: each sentence is its own clause
            return [[i] for i in range(len(sentences))]
    
    def _merge_adjacent_similar(self, 
                                 clusters: List[List[int]],
                                 embeddings: np.ndarray,
                                 threshold: float = 0.7) -> List[List[int]]:
        """
        Merge adjacent clusters if they are similar
        """
        if len(clusters) <= 1:
            return clusters
        
        merged = []
        current = clusters[0]
        
        for next_cluster in clusters[1:]:
            # Get mean embeddings
            current_emb = np.mean(embeddings[current], axis=0)
            next_emb = np.mean(embeddings[next_cluster], axis=0)
            
            # Check similarity
            sim = cosine_similarity([current_emb], [next_emb])[0][0]
            
            if sim > threshold:
                # Merge
                current = current + next_cluster
            else:
                merged.append(current)
                current = next_cluster
        
        merged.append(current)
        return merged
    
    def segment(self, text: str) -> List[Clause]:
        """
        Segment document into clauses
        
        Args:
            text: Document text
            
        Returns:
            List of Clause objects
        """
        print("\n🔍 Enhanced Clause Segmentation")
        print(f"   Similarity threshold: {self.similarity_threshold}")
        
        # Check for numbered clauses first
        numbered = self._detect_numbered_clauses(text)
        if numbered and len(numbered) >= 3:
            print(f"   📋 Detected {len(numbered)} numbered clauses")
            return self._process_numbered_clauses(numbered)
        
        # Split into sentences
        sentences = self._split_sentences(text)
        print(f"   📝 Found {len(sentences)} sentences")
        
        if not sentences:
            return []
        
        if self.embedding_model is None:
            # Fallback: simple segmentation
            return self._simple_segment(sentences)
        
        # Get embeddings
        print("   🧠 Computing sentence embeddings...")
        embeddings = self.embedding_model.encode(sentences)
        
        # Cluster sentences
        print("   🔗 Clustering sentences...")
        clusters = self._cluster_sentences(sentences, embeddings)
        
        # Merge adjacent similar clusters
        clusters = self._merge_adjacent_similar(clusters, embeddings)
        print(f"   📊 Created {len(clusters)} clauses")
        
        # Build clause objects
        clauses = []
        for i, cluster_indices in enumerate(clusters):
            cluster_sentences = [sentences[idx] for idx in cluster_indices]
            content = ' '.join(cluster_sentences)
            
            # Classify clause type
            clause_type, confidence = self._classify_clause_type(content)
            
            # Extract keywords
            keywords = self._extract_keywords(content)
            
            clause = Clause(
                id=i + 1,
                clause_type=clause_type,
                title=f"Clause {i + 1}: {clause_type.value.title()}",
                content=content,
                sentences=cluster_sentences,
                start_idx=min(cluster_indices),
                end_idx=max(cluster_indices),
                confidence=confidence,
                risk_score=0.0,  # Will be set by risk detector
                keywords=keywords
            )
            clauses.append(clause)
        
        return clauses
    
    def _process_numbered_clauses(self, numbered: List[Tuple[int, str, str]]) -> List[Clause]:
        """Process pre-numbered clauses"""
        clauses = []
        
        for i, (num, content, _) in enumerate(numbered):
            clause_type, confidence = self._classify_clause_type(content)
            keywords = self._extract_keywords(content)
            
            clause = Clause(
                id=num,
                clause_type=clause_type,
                title=f"Clause {num}: {clause_type.value.title()}",
                content=content,
                sentences=[content],
                start_idx=i,
                end_idx=i,
                confidence=confidence,
                keywords=keywords
            )
            clauses.append(clause)
        
        return clauses
    
    def _simple_segment(self, sentences: List[str]) -> List[Clause]:
        """Fallback simple segmentation"""
        clauses = []
        
        for i, sent in enumerate(sentences):
            clause = Clause(
                id=i + 1,
                clause_type=ClauseType.GENERAL,
                title=f"Clause {i + 1}",
                content=sent,
                sentences=[sent],
                start_idx=i,
                end_idx=i,
                confidence=0.5
            )
            clauses.append(clause)
        
        return clauses
    
    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract important keywords from clause"""
        # Important legal terms
        legal_terms = [
            'rent', 'deposit', 'security', 'tenant', 'landlord', 'lessor', 'lessee',
            'termination', 'notice', 'penalty', 'maintenance', 'repair', 'sublet',
            'agreement', 'premises', 'possession', 'breach', 'liable', 'rupees',
            'monthly', 'advance', 'refund', 'electricity', 'water', 'charges'
        ]
        
        text_lower = text.lower()
        found = []
        
        for term in legal_terms:
            if term in text_lower:
                found.append(term)
        
        return found[:top_k]


# Backward compatibility
MLClauseSegmenter = EnhancedClauseSegmenter


if __name__ == "__main__":
    # Test
    segmenter = EnhancedClauseSegmenter()
    
    test_text = """
    This rental agreement is made between Mr. Ramesh Kumar (Landlord) and 
    Mr. Suresh Sharma (Tenant) on 1st January 2025.
    
    1. The monthly rent shall be Rs. 25,000 payable on or before the 5th of every month.
    
    2. The tenant has paid Rs. 1,50,000 as security deposit which will be refunded 
    at the time of vacating the premises.
    
    3. The agreement is for a period of 11 months from the date of commencement.
    
    4. Either party may terminate this agreement by giving 2 months written notice.
    
    5. The tenant shall not sublet or transfer the premises to any third party.
    
    6. The tenant shall pay electricity and water charges at actuals.
    """
    
    clauses = segmenter.segment(test_text)
    
    for clause in clauses:
        print(f"\n{clause.title}")
        print(f"   Type: {clause.clause_type.value}")
        print(f"   Confidence: {clause.confidence:.2f}")
        print(f"   Keywords: {clause.keywords}")
        print(f"   Content: {clause.content[:100]}...")
