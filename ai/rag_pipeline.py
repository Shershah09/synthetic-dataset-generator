"""Retrieval-Augmented Generation (RAG) Pipeline for Synthetic Dataset Generator.

Implements the 8-stage RAG architecture described in Chapter 3.1.4 and 4.2.4:
1. Document Ingestion
2. Text Extraction & Cleaning
3. Document Chunking
4. Embedding Generation
5. Vector Storage & Metadata
6. Query Processing
7. Similarity Search (Cosine similarity)
8. Context Augmentation for Schema Generation
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DocumentChunk:
    """Represents a chunk of domain knowledge in the vector store."""
    chunk_id: str
    domain: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[Dict[str, float]] = None


# =============================================================================
# Domain Knowledge Base Documents
# =============================================================================
RAW_KNOWLEDGE_DOCUMENTS = [
    {
        "domain": "Healthcare",
        "title": "Clinical Cardiovascular & Metabolic Reference Ranges",
        "text": """
        In clinical medicine and epidemiology, vital signs and lab values follow specific physiological bounds:
        1. Blood Pressure: Systolic BP ranges from 90 to 180 mmHg. Normal is < 120 mmHg, Stage 1 hypertension is 130-139, Stage 2 is >= 140 mmHg.
        2. Body Mass Index (BMI): Ranges from 16.0 to 45.0 kg/m2. Underweight is < 18.5, Normal is 18.5-24.9, Overweight is 25.0-29.9, Obese is >= 30.0.
        3. Total Cholesterol: Ranges from 130 to 320 mg/dL. Desirable is < 200, Borderline high is 200-239, High is >= 240 mg/dL.
        4. Fasting Blood Glucose: Ranges from 70 to 260 mg/dL. Normal is 70-99, Prediabetes is 100-125, Diabetes is >= 126 mg/dL.
        5. Age & Disease Risk: Cardiovascular disease probability increases non-linearly with Age (>50), high BMI, Smoking status, and Elevated Systolic Blood Pressure.
        6. Missing Values Policy: Patient_ID must never have missing values. Clinical lab tests may have 2-8% random missingness.
        """
    },
    {
        "domain": "Banking and Finance",
        "title": "Consumer Credit Risk & Basel Regulatory Standards",
        "text": """
        Banking risk assessment and loan default prediction adhere to standardized financial parameters:
        1. Credit Score: FICO score ranges from 300 to 850. Poor is 300-579, Fair is 580-669, Good is 670-739, Very Good is 740-799, Exceptional is 800-850.
        2. Debt-to-Income (DTI) Ratio: Ranges from 0.05 to 0.70. Ratios above 0.43 represent high default risk under qualified mortgage standards.
        3. Annual Income: Follows log-normal distribution, typically ranging from $15,000 to $250,000+.
        4. Loan Amount: Correlates positively with Annual Income, typically $2,000 to $100,000.
        5. Default Relationship: High DTI, lower credit score (<620), and past delinquencies drastically increase loan default probability.
        6. Class Imbalance: Credit default datasets in retail banking typically have 15-25% positive default rates.
        """
    },
    {
        "domain": "E-commerce",
        "title": "Customer Lifetime Value & Churn Analytics",
        "text": """
        E-commerce behavioral analytics revolves around RFM (Recency, Frequency, Monetary) modeling:
        1. Days Since Last Login (Recency): Ranges from 0 to 180+ days. Inactive users (>60 days) have an exponential increase in churn probability.
        2. Total Purchases (Frequency): Discrete count typically 1 to 150 transactions. Poisson distributed.
        3. Average Order Value (AOV - Monetary): Log-normal distribution, ranging from $10 to $500.
        4. Customer Satisfaction Score (CSAT): Ordinal scale 1 to 5. Scores <= 2 correlate heavily with user attrition.
        5. Discount Usage Ratio: Bounded between 0.0 and 1.0. High discount reliance often signals price sensitivity.
        6. Target Imbalance: Typical SaaS and e-commerce churn rates range between 18% and 30%.
        """
    },
    {
        "domain": "Education",
        "title": "Academic Performance & Student Learning Metrics",
        "text": """
        Educational data science and student retention modeling use behavioral and assessment features:
        1. Grade Point Average (GPA): Continuous scale from 1.0 to 4.0. Strong predictor of exam outcomes.
        2. Weekly Study Hours: Bounded between 2 and 40 hours. Diminishing returns observed beyond 30 hours per week.
        3. Attendance Rate: Expressed as percentage 45% to 100%. Mandatory threshold is often 75%. Attendance strongly correlates with GPA.
        4. Sleep Duration: Ranges from 4 to 10 hours daily. Extreme deprivation (<5 hours) impairs exam performance.
        5. Target Variable: Final Exam Score (continuous 0-100 for regression) or Pass/Fail (classification with 65-75% pass rate).
        """
    },
    {
        "domain": "Agriculture",
        "title": "Precision Farming, Soil Chemistry & Crop Yield Standards",
        "text": """
        Agronomic datasets model soil nutrients, environmental precipitation, and vegetative yield:
        1. Soil pH: Bounded between 4.5 (strongly acidic) and 8.5 (strongly alkaline). Optimal neutral zone is 6.0 to 7.2.
        2. Soil Moisture: Percentage volume ranging from 10% to 75%. Severe drought stress occurs below 20%.
        3. Macro-nutrients (N-P-K): Nitrogen (15-140 mg/kg), Phosphorus (10-90 mg/kg), Potassium (10-150 mg/kg).
        4. Temperature & Rainfall: Temperature (12-42 deg C), Annual/Seasonal rainfall (30-320 mm).
        5. Crop Yield Target: Metric tons per hectare (continuous 1.0 to 15.0 tons) or Suitability rating (Low/Medium/High).
        """
    },
    {
        "domain": "General ML",
        "title": "Machine Learning Benchmark Standards & Covariance Modeling",
        "text": """
        General ML synthetic data benchmarking requirements:
        1. Feature Scaling: Standard normal features centered at 0 with standard deviation 1 to 3.
        2. Correlation Modeling: Controlled covariance matrices to simulate realistic multicollinearity without singularity.
        3. Noise Injection: Gaussian noise with configurable sigma (0.05 to 0.40) to test model robustness and prevent trivial overfitting.
        4. Target Construction: Logistic latent scores for classification; linear combinations with interaction terms for regression.
        5. Clustering Centroids: Separable multi-modal cluster centroids (k=2 to 8) with distinct spatial shifts.
        """
    }
]


# =============================================================================
# Vector Store & Embedding Engine
# =============================================================================
class RAGPipeline:
    """In-memory Vector Database and Retrieval-Augmented Generation engine."""

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._ingest_and_index()

    def _tokenize(self, text: str) -> List[str]:
        """Normalize and tokenize text into lowercase word tokens."""
        clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        tokens = [t for t in clean.split() if len(t) > 2]
        return tokens

    def _ingest_and_index(self):
        """Chunk raw documents, compute TF-IDF embeddings, and store in vector index."""
        # 1. Chunking
        chunk_idx = 1
        for doc in RAW_KNOWLEDGE_DOCUMENTS:
            paragraphs = [p.strip() for p in doc["text"].split("\n") if len(p.strip()) > 30]
            for p in paragraphs:
                chunk = DocumentChunk(
                    chunk_id=f"chunk_{chunk_idx:03d}",
                    domain=doc["domain"],
                    title=doc["title"],
                    content=p,
                    metadata={"domain": doc["domain"], "source": doc["title"]},
                )
                self.chunks.append(chunk)
                chunk_idx += 1

        # 2. Build Vocabulary & IDF
        doc_count = len(self.chunks)
        doc_freq: Dict[str, int] = {}

        for chk in self.chunks:
            words = set(self._tokenize(chk.content))
            for w in words:
                doc_freq[w] = doc_freq.get(w, 0) + 1

        self.idf = {
            w: math.log((doc_count + 1.0) / (df + 1.0)) + 1.0
            for w, df in doc_freq.items()
        }

        # 3. Compute Embeddings for Chunks
        for chk in self.chunks:
            tokens = self._tokenize(chk.content)
            tf: Dict[str, float] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0.0) + 1.0

            vec: Dict[str, float] = {}
            norm_sq = 0.0
            for t, count in tf.items():
                tfidf_val = count * self.idf.get(t, 1.0)
                vec[t] = tfidf_val
                norm_sq += tfidf_val ** 2

            # Normalize vector
            norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
            chk.vector = {t: val / norm for t, val in vec.items()}

    def _embed_query(self, query: str) -> Dict[str, float]:
        """Convert query string into normalized TF-IDF embedding vector."""
        tokens = self._tokenize(query)
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0.0) + 1.0

        vec: Dict[str, float] = {}
        norm_sq = 0.0
        for t, count in tf.items():
            tfidf_val = count * self.idf.get(t, 1.0)
            vec[t] = tfidf_val
            norm_sq += tfidf_val ** 2

        norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
        return {t: val / norm for t, val in vec.items()}

    def similarity_search(
        self, query: str, domain: Optional[str] = None, top_k: int = 3
    ) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve top_k chunks matching query with cosine similarity scores."""
        query_vec = self._embed_query(query)
        results: List[Tuple[DocumentChunk, float]] = []

        for chk in self.chunks:
            if domain and domain != "General ML" and chk.domain != domain and chk.domain != "General ML":
                continue

            if not chk.vector:
                continue

            # Compute Cosine Dot Product
            score = 0.0
            for term, q_val in query_vec.items():
                if term in chk.vector:
                    score += q_val * chk.vector[term]

            # Domain keyword boost if query explicitly mentions domain
            if domain and chk.domain.lower() in query.lower():
                score += 0.15

            if score > 0.01:
                results.append((chk, round(score, 4)))

        # Sort descending by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def build_augmented_context(self, prompt: str, domain: Optional[str] = None) -> str:
        """Create structured context string to augment the LLM prompt."""
        matches = self.similarity_search(prompt, domain=domain, top_k=3)
        if not matches:
            return "Domain knowledge reference: Standard statistical bounds and distributions apply."

        context_lines = ["--- RETRIEVED DOMAIN KNOWLEDGE (RAG CONTEXT) ---"]
        for idx, (chunk, score) in enumerate(matches, 1):
            context_lines.append(f"[{idx}] (Domain: {chunk.domain}, Relevance: {score:.2f})")
            context_lines.append(f"Title: {chunk.title}")
            context_lines.append(f"Rule: {chunk.content.strip()}\n")
        context_lines.append("--------------------------------------------------")

        return "\n".join(context_lines)


# Global singleton pipeline instance
_rag_instance: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create singleton RAG pipeline instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGPipeline()
    return _rag_instance
