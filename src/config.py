from dataclasses import dataclass, field
from src.utils import LLM_Model
@dataclass
class LinearRAGConfig:
    dataset_name: str
    embedding_model: str = "all-mpnet-base-v2"
    llm_model: LLM_Model = None
    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100
    spacy_model: str = "en_core_web_sm"
    spacy_fallback_model: str = "en_core_web_trf"
    working_dir: str = "./import"
    batch_size: int = 128
    max_workers: int = 16
    retrieval_top_k: int = 5
    max_iterations: int = 3
    top_k_sentence: int = 1
    passage_ratio: float = 1.5
    passage_node_weight: float = 0.05
    damping: float = 0.5
    iteration_threshold: float = 0.5
    use_vectorized_retrieval: bool = False  # True for vectorized matrix computation, False for BFS iteration
    enable_hybrid_attribute_fallback: bool = False
    attribute_keyword_boost: float = 0.25
    enable_structured_memory_graph: bool = True
    structured_node_top_k: int = 24
    structured_node_weight: float = 0.35
    structured_event_weight: float = 1.0
    structured_memory_weight: float = 0.9
    structured_time_weight: float = 0.8
    structured_score_threshold: float = 0.2
    graphml_write_interval: int = 5
    attribute_query_keywords: list[str] = field(default_factory=lambda: [
        "born", "birth", "where", "when", "located", "location", "founded", "founder",
        "died", "death", "nationality", "capital", "date", "year"
    ])
