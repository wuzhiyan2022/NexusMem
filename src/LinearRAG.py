from src.embedding_store import EmbeddingStore
from src.utils import min_max_normalize
import os
import json
from collections import defaultdict
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.ner import SpacyNER
import igraph as ig
import re
import logging
import torch
logger = logging.getLogger(__name__)


class LinearRAG:
    def __init__(self, global_config):
        self.config = global_config
        logger.info(f"Initializing LinearRAG with config: {self.config}")
        retrieval_method = "Vectorized Matrix-based" if self.config.use_vectorized_retrieval else "BFS Iteration"
        logger.info(f"Using retrieval method: {retrieval_method}")
        
        # Setup device for GPU acceleration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.config.use_vectorized_retrieval:
            logger.info(f"Using device: {self.device} for vectorized retrieval")
        
        self.dataset_name = global_config.dataset_name
        self.load_embedding_store()
        self.llm_model = self.config.llm_model
        self.spacy_ner = getattr(self.config, "spacy_ner", None) or SpacyNER(self.config.spacy_model)
        self.graph = ig.Graph(directed=False)
        self.structured_node_texts = {}
        self.structured_node_types = {}
        self.structured_node_status = {}
        self.structured_node_ids = []
        self.structured_node_embeddings = None

    def load_embedding_store(self):
        self.passage_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "passage_embedding.parquet"), batch_size=self.config.batch_size, namespace="passage")
        self.entity_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "entity_embedding.parquet"), batch_size=self.config.batch_size, namespace="entity")
        self.sentence_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "sentence_embedding.parquet"), batch_size=self.config.batch_size, namespace="sentence")

    def load_existing_data(self,passage_hash_ids):
        self.ner_results_path = os.path.join(self.config.working_dir,self.dataset_name, "ner_results.json")
        if os.path.exists(self.ner_results_path):
            existing_ner_reuslts = json.load(open(self.ner_results_path))
            existing_passage_hash_id_to_entities = existing_ner_reuslts["passage_hash_id_to_entities"]
            existing_sentence_to_entities = existing_ner_reuslts["sentence_to_entities"]
            existing_passage_hash_ids = set(existing_passage_hash_id_to_entities.keys())
            new_passage_hash_ids = set(passage_hash_ids) - existing_passage_hash_ids
            return existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids
        else:
            return {}, {}, passage_hash_ids

    def qa(self, questions):
        retrieval_results = self.retrieve(questions)
        system_prompt = f"""As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations."""
        all_messages = []
        for retrieval_result in retrieval_results:
            question = retrieval_result["question"]
            sorted_passage = retrieval_result["sorted_passage"]
            prompt_user = """"""
            for passage in sorted_passage:
                prompt_user += f"{passage}\n"
            prompt_user += f"Question: {question}\n Thought: "
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_user}
            ]
            all_messages.append(messages)
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            all_qa_results = list(tqdm(
                executor.map(self.llm_model.infer, all_messages),
                total=len(all_messages),
                desc="QA Reading (Parallel)"
            ))

        for qa_result,question_info in zip(all_qa_results,retrieval_results):
            try:
                pred_ans = qa_result.split('Answer:')[1].strip()
            except:
                pred_ans = qa_result
            question_info["pred_answer"] = pred_ans
        return retrieval_results
        
    def retrieve(self, questions):
        self.entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self.entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        self.passage_hash_ids = list(self.passage_embedding_store.hash_id_to_text.keys())
        self.passage_embeddings = np.array(self.passage_embedding_store.embeddings)
        self.sentence_hash_ids = list(self.sentence_embedding_store.hash_id_to_text.keys())
        self.sentence_embeddings = np.array(self.sentence_embedding_store.embeddings)
        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}
        self.vertex_idx_to_node_name = {v.index: v["name"] for v in self.graph.vs if "name" in v.attributes()}

        # Precompute sparse matrices for vectorized retrieval if needed
        if self.config.use_vectorized_retrieval:
            logger.info("Precomputing sparse adjacency matrices for vectorized retrieval...")
            self._precompute_sparse_matrices()
            e2s_shape = self.entity_to_sentence_sparse.shape
            s2e_shape = self.sentence_to_entity_sparse.shape
            e2s_nnz = self.entity_to_sentence_sparse._nnz()
            s2e_nnz = self.sentence_to_entity_sparse._nnz()
            logger.info(f"Matrices built: Entity-Sentence {e2s_shape}, Sentence-Entity {s2e_shape}")
            logger.info(f"E2S Sparsity: {(1 - e2s_nnz / (e2s_shape[0] * e2s_shape[1])) * 100:.2f}% (nnz={e2s_nnz})")
            logger.info(f"S2E Sparsity: {(1 - s2e_nnz / (s2e_shape[0] * s2e_shape[1])) * 100:.2f}% (nnz={s2e_nnz})")
            logger.info(f"Device: {self.device}")

        retrieval_results = []
        for question_info in tqdm(questions, desc="Retrieving"):
            question = question_info["question"]
            question_embedding = self.config.embedding_model.encode(question,normalize_embeddings=True,show_progress_bar=False,batch_size=self.config.batch_size)
            seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores = self.get_seed_entities(question)
            structured_node_weights = self.calculate_structured_node_scores(question, question_embedding)
            has_structured_seeds = bool(np.any(structured_node_weights > 0))
            if len(seed_entities) != 0 or has_structured_seeds:
                sorted_passage_hash_ids,sorted_passage_scores = self.graph_search_multi_entry(
                    question,
                    question_embedding,
                    seed_entity_indices,
                    seed_entities,
                    seed_entity_hash_ids,
                    seed_entity_scores,
                    structured_node_weights,
                )
                final_passage_hash_ids = sorted_passage_hash_ids[:self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                final_passages = [self.passage_embedding_store.hash_id_to_text[passage_hash_id] for passage_hash_id in final_passage_hash_ids]
            else:
                sorted_passage_indices,sorted_passage_scores = self.dense_passage_retrieval(question_embedding)
                final_passage_indices = sorted_passage_indices[:self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                final_passages = [self.passage_embedding_store.texts[idx] for idx in final_passage_indices]
            result = {
                "question": question,
                "sorted_passage": final_passages,
                "sorted_passage_scores": final_passage_scores,
                "gold_answer": question_info["answer"]
            }
            retrieval_results.append(result)
        return retrieval_results
    
    def _precompute_sparse_matrices(self):
        """
        Precompute and cache sparse adjacency matrices for efficient vectorized retrieval using PyTorch.
        This is called once at the beginning of retrieve() to avoid rebuilding matrices per query.
        """
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)
        
        # Build entity-to-sentence matrix (Mention matrix) using COO format
        entity_to_sentence_indices = []
        entity_to_sentence_values = []
        
        for entity_hash_id, sentence_hash_ids in self.entity_hash_id_to_sentence_hash_ids.items():
            entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
            for sentence_hash_id in sentence_hash_ids:
                sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
                entity_to_sentence_indices.append([entity_idx, sentence_idx])
                entity_to_sentence_values.append(1.0)
        
        # Build sentence-to-entity matrix
        sentence_to_entity_indices = []
        sentence_to_entity_values = []
        
        for sentence_hash_id, entity_hash_ids in self.sentence_hash_id_to_entity_hash_ids.items():
            sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
            for entity_hash_id in entity_hash_ids:
                entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
                sentence_to_entity_indices.append([sentence_idx, entity_idx])
                sentence_to_entity_values.append(1.0)
        
        # Convert to PyTorch sparse tensors (COO format, then convert to CSR for efficiency)
        if len(entity_to_sentence_indices) > 0:
            e2s_indices = torch.tensor(entity_to_sentence_indices, dtype=torch.long).t()
            e2s_values = torch.tensor(entity_to_sentence_values, dtype=torch.float32)
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                e2s_indices, e2s_values, (num_entities, num_sentences), device=self.device
            ).coalesce()
        else:
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_entities, num_sentences), device=self.device
            )
        
        if len(sentence_to_entity_indices) > 0:
            s2e_indices = torch.tensor(sentence_to_entity_indices, dtype=torch.long).t()
            s2e_values = torch.tensor(sentence_to_entity_values, dtype=torch.float32)
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                s2e_indices, s2e_values, (num_sentences, num_entities), device=self.device
            ).coalesce()
        else:
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_sentences, num_entities), device=self.device
            )
            
    def graph_search_with_seed_entities(self, question, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        if self.config.use_vectorized_retrieval:
            entity_weights, actived_entities = self.calculate_entity_scores_vectorized(question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores)
        else:
            entity_weights, actived_entities = self.calculate_entity_scores(question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores)
        passage_weights = self.calculate_passage_scores(question, question_embedding, actived_entities)
        node_weights = entity_weights + passage_weights
        ppr_sorted_passage_indices,ppr_sorted_passage_scores = self.run_ppr(node_weights)
        return ppr_sorted_passage_indices,ppr_sorted_passage_scores

    def graph_search_multi_entry(
        self,
        question,
        question_embedding,
        seed_entity_indices,
        seed_entities,
        seed_entity_hash_ids,
        seed_entity_scores,
        structured_node_weights,
    ):
        if len(seed_entities) != 0:
            if self.config.use_vectorized_retrieval:
                entity_weights, actived_entities = self.calculate_entity_scores_vectorized(
                    question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
                )
            else:
                entity_weights, actived_entities = self.calculate_entity_scores(
                    question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
                )
        else:
            entity_weights = np.zeros(len(self.graph.vs["name"]))
            actived_entities = {}
        passage_weights = self.calculate_passage_scores(question, question_embedding, actived_entities)
        node_weights = entity_weights + passage_weights + structured_node_weights
        ppr_sorted_passage_indices,ppr_sorted_passage_scores = self.run_ppr(node_weights)
        return ppr_sorted_passage_indices,ppr_sorted_passage_scores

    def run_ppr(self, node_weights):        
        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=self.config.damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )
        
        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_indices])
        sorted_indices_in_doc_scores = np.argsort(doc_scores)[::-1]
        sorted_passage_scores = doc_scores[sorted_indices_in_doc_scores]
        
        sorted_passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[i]] 
            for i in sorted_indices_in_doc_scores
        ]
        
        return sorted_passage_hash_ids, sorted_passage_scores.tolist()

    def calculate_entity_scores(self,question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
        actived_entities = {}
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        for seed_entity_idx,seed_entity,seed_entity_hash_id,seed_entity_score in zip(seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 1)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score    
        used_sentence_hash_ids = set()
        current_entities = actived_entities.copy()
        iteration = 1
        while len(current_entities) > 0 and iteration < self.config.max_iterations:
            new_entities = {}
            for entity_hash_id, (entity_id, entity_score, tier) in current_entities.items():
                if entity_score < self.config.iteration_threshold:
                    continue
                sentence_hash_ids = [sid for sid in list(self.entity_hash_id_to_sentence_hash_ids[entity_hash_id]) if sid not in used_sentence_hash_ids]
                if not sentence_hash_ids:
                    continue
                sentence_indices = [self.sentence_embedding_store.hash_id_to_idx[sid] for sid in sentence_hash_ids]
                sentence_embeddings = self.sentence_embeddings[sentence_indices]
                question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
                sentence_similarities = np.dot(sentence_embeddings, question_emb).flatten()
                top_sentence_indices = np.argsort(sentence_similarities)[::-1][:self.config.top_k_sentence]
                for top_sentence_index in top_sentence_indices:
                    top_sentence_hash_id = sentence_hash_ids[top_sentence_index]
                    top_sentence_score = sentence_similarities[top_sentence_index]
                    used_sentence_hash_ids.add(top_sentence_hash_id)
                    entity_hash_ids_in_sentence = self.sentence_hash_id_to_entity_hash_ids[top_sentence_hash_id]
                    for next_entity_hash_id in entity_hash_ids_in_sentence:
                        next_entity_score = entity_score * top_sentence_score
                        if next_entity_score < self.config.iteration_threshold:
                            continue
                        next_enitity_node_idx = self.node_name_to_vertex_idx[next_entity_hash_id]
                        entity_weights[next_enitity_node_idx] += next_entity_score
                        new_entities[next_entity_hash_id] = (next_enitity_node_idx, next_entity_score, iteration+1)
            actived_entities.update(new_entities)
            current_entities = new_entities.copy()
            iteration += 1
        return entity_weights, actived_entities

    def calculate_entity_scores_vectorized(self,question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
        """
        GPU-accelerated vectorized version using PyTorch sparse tensors.
        Uses sparse representation for both matrices and entity score vectors for maximum efficiency.
        Now includes proper dynamic pruning to match BFS behavior:
        - Sentence deduplication (tracks used sentences)
        - Per-entity top-k sentence selection
        - Proper threshold-based pruning
        """
        # Initialize entity weights
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)
        
        # Compute all sentence similarities with the question at once
        question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
        sentence_similarities_np = np.dot(self.sentence_embeddings, question_emb).flatten()
        
        # Convert to torch tensors and move to device
        sentence_similarities = torch.from_numpy(sentence_similarities_np).float().to(self.device)
        
        # Track used sentences for deduplication (like BFS version)
        used_sentence_mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.device)
        
        # Initialize seed entity scores as sparse tensor
        seed_indices = torch.tensor([[idx] for idx in seed_entity_indices], dtype=torch.long).t()
        seed_values = torch.tensor(seed_entity_scores, dtype=torch.float32)
        entity_scores_sparse = torch.sparse_coo_tensor(
            seed_indices, seed_values, (num_entities,), device=self.device
        ).coalesce()
        
        # Also maintain a dense accumulator for total scores
        entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.device)
        entity_scores_dense.scatter_(0, torch.tensor(seed_entity_indices, device=self.device), 
                                     torch.tensor(seed_entity_scores, dtype=torch.float32, device=self.device))
        
        # Initialize actived_entities
        actived_entities = {}
        for seed_entity_idx, seed_entity, seed_entity_hash_id, seed_entity_score in zip(
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
        ):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 0)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score
        
        current_entity_scores_sparse = entity_scores_sparse
        
        # Iterative matrix-based propagation using sparse matrices on GPU
        for iteration in range(1, self.config.max_iterations):
            # Convert sparse tensor to dense for threshold operation
            current_entity_scores_dense = current_entity_scores_sparse.to_dense()
            
            # Apply threshold to current scores
            current_entity_scores_dense = torch.where(
                current_entity_scores_dense >= self.config.iteration_threshold, 
                current_entity_scores_dense, 
                torch.zeros_like(current_entity_scores_dense)
            )
            
            # Get non-zero indices for sparse representation
            nonzero_mask = current_entity_scores_dense > 0
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(-1)
            
            if len(nonzero_indices) == 0:
                break
            
            # Extract non-zero values and create sparse tensor
            nonzero_values = current_entity_scores_dense[nonzero_indices]
            current_entity_scores_sparse = torch.sparse_coo_tensor(
                nonzero_indices.unsqueeze(0), nonzero_values, (num_entities,), device=self.device
            ).coalesce()
            
            # Step 1: Sparse entity scores @ Sparse E2S matrix
            # Convert sparse vector to 2D for matrix multiplication
            current_scores_2d = torch.sparse_coo_tensor(
                torch.stack([nonzero_indices, torch.zeros_like(nonzero_indices)]),
                nonzero_values,
                (num_entities, 1),
                device=self.device
            ).coalesce()
            
            # E @ E2S -> sentence activation scores (sparse @ sparse = dense)
            sentence_activation = torch.sparse.mm(
                self.entity_to_sentence_sparse.t(),
                current_scores_2d
            )
            # Convert to dense before squeeze to avoid CUDA sparse tensor issues
            if sentence_activation.is_sparse:
                sentence_activation = sentence_activation.to_dense()
            sentence_activation = sentence_activation.squeeze()
            
            # Apply sentence deduplication: mask out used sentences
            sentence_activation = torch.where(
                used_sentence_mask,
                torch.zeros_like(sentence_activation),
                sentence_activation
            )
            
            # Step 2: Per-entity top-k sentence selection
            # This matches BFS behavior: each entity independently selects its top-k sentences
            selected_sentence_indices_list = []
            
            if len(nonzero_indices) > 0 and self.config.top_k_sentence > 0:
                # Iterate through each active entity
                for i, entity_idx in enumerate(nonzero_indices):
                    entity_score = nonzero_values[i]
                    
                    # Get sentences connected to this entity from the sparse matrix
                    # entity_to_sentence_sparse shape: (num_entities, num_sentences)
                    entity_row = self.entity_to_sentence_sparse[entity_idx].coalesce()
                    entity_sentence_indices = entity_row.indices()[0]  # Get column indices
                    
                    if len(entity_sentence_indices) == 0:
                        continue
                    
                    # Filter out already used sentences
                    sentence_mask = ~used_sentence_mask[entity_sentence_indices]
                    available_sentence_indices = entity_sentence_indices[sentence_mask]
                    
                    if len(available_sentence_indices) == 0:
                        continue
                    
                    # Get sentence similarities (for ranking)
                    sentence_sims = sentence_similarities[available_sentence_indices]
                    
                    # Select top-k sentences based ONLY on sentence similarity (matches BFS line 240)
                    # NOT weighted by entity_score at selection time
                    k = min(self.config.top_k_sentence, len(sentence_sims))
                    if k > 0:
                        top_k_values, top_k_local_indices = torch.topk(sentence_sims, k)
                        top_k_sentence_indices = available_sentence_indices[top_k_local_indices]
                        selected_sentence_indices_list.append(top_k_sentence_indices)
                
                # Merge all selected sentences (with deduplication via unique)
                if len(selected_sentence_indices_list) > 0:
                    all_selected_sentences = torch.cat(selected_sentence_indices_list)
                    unique_selected_sentences = torch.unique(all_selected_sentences)
                    
                    # Mark selected sentences as used
                    used_sentence_mask[unique_selected_sentences] = True
                    
                    # Compute weighted sentence scores for propagation
                    # weighted_score = sentence_activation * sentence_similarity
                    weighted_sentence_scores = sentence_activation * sentence_similarities
                    
                    # Zero out non-selected sentences
                    mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.device)
                    mask[unique_selected_sentences] = True
                    weighted_sentence_scores = torch.where(
                        mask,
                        weighted_sentence_scores,
                        torch.zeros_like(weighted_sentence_scores)
                    )
                else:
                    # No sentences selected, create zero vector
                    weighted_sentence_scores = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)
            else:
                # No active entities or top_k_sentence is 0
                weighted_sentence_scores = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)
            
            # Step 3: Weighted sentences @ S2E -> propagate to next entities
            # Convert to sparse for more efficient computation
            weighted_nonzero_mask = weighted_sentence_scores > 0
            weighted_nonzero_indices = torch.nonzero(weighted_nonzero_mask, as_tuple=False).squeeze(-1)
            
            if len(weighted_nonzero_indices) > 0:
                weighted_nonzero_values = weighted_sentence_scores[weighted_nonzero_indices]
                weighted_scores_2d = torch.sparse_coo_tensor(
                    torch.stack([weighted_nonzero_indices, torch.zeros_like(weighted_nonzero_indices)]),
                    weighted_nonzero_values,
                    (num_sentences, 1),
                    device=self.device
                ).coalesce()
                
                next_entity_scores_result = torch.sparse.mm(
                    self.sentence_to_entity_sparse.t(),
                    weighted_scores_2d
                )
                # Convert to dense before squeeze to avoid CUDA sparse tensor issues
                if next_entity_scores_result.is_sparse:
                    next_entity_scores_result = next_entity_scores_result.to_dense()
                next_entity_scores_dense = next_entity_scores_result.squeeze()
            else:
                next_entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.device)
            
            # Update entity scores (accumulate in dense format)
            entity_scores_dense += next_entity_scores_dense
            
            # Update actived_entities dictionary (record last trigger like BFS)
            # This matches BFS behavior: unconditionally update for entities above threshold
            next_entity_scores_np = next_entity_scores_dense.cpu().numpy()
            active_indices = np.where(next_entity_scores_np >= self.config.iteration_threshold)[0]
            for entity_idx in active_indices:
                score = next_entity_scores_np[entity_idx]
                entity_hash_id = self.entity_hash_ids[entity_idx]
                # Unconditionally update to record the last trigger (matches BFS line 252)
                actived_entities[entity_hash_id] = (entity_idx, float(score), iteration)
            
            # Prepare sparse tensor for next iteration
            next_nonzero_mask = next_entity_scores_dense > 0
            next_nonzero_indices = torch.nonzero(next_nonzero_mask, as_tuple=False).squeeze(-1)
            if len(next_nonzero_indices) > 0:
                next_nonzero_values = next_entity_scores_dense[next_nonzero_indices]
                current_entity_scores_sparse = torch.sparse_coo_tensor(
                    next_nonzero_indices.unsqueeze(0), next_nonzero_values, 
                    (num_entities,), device=self.device
                ).coalesce()
            else:
                break
        
        # Convert back to numpy for final processing
        entity_scores_final = entity_scores_dense.cpu().numpy()
        
        # Map entity scores to graph node weights (only for non-zero scores)
        nonzero_indices = np.where(entity_scores_final > 0)[0]
        for entity_idx in nonzero_indices:
            score = entity_scores_final[entity_idx]
            entity_hash_id = self.entity_hash_ids[entity_idx]
            entity_node_idx = self.node_name_to_vertex_idx[entity_hash_id]
            entity_weights[entity_node_idx] = float(score)
        
        return entity_weights, actived_entities

    def calculate_passage_scores(self, question, question_embedding, actived_entities):
        passage_weights = np.zeros(len(self.graph.vs["name"]))
        dpr_passage_indices, dpr_passage_scores = self.dense_passage_retrieval(question_embedding)
        dpr_passage_scores = min_max_normalize(dpr_passage_scores)
        apply_attribute_boost = (
            self.config.enable_hybrid_attribute_fallback
            and self._is_attribute_query(question)
        )
        question_lower = question.lower()

        for i, dpr_passage_index in enumerate(dpr_passage_indices):
            total_entity_bonus = 0
            passage_hash_id = self.passage_embedding_store.hash_ids[dpr_passage_index]
            dpr_passage_score = dpr_passage_scores[i]
            passage_text_lower = self._visible_passage_text(
                self.passage_embedding_store.hash_id_to_text[passage_hash_id]
            ).lower()
            for entity_hash_id, (entity_id, entity_score, tier) in actived_entities.items():
                entity_lower = self.entity_embedding_store.hash_id_to_text[entity_hash_id].lower()
                entity_occurrences = passage_text_lower.count(entity_lower)
                if entity_occurrences > 0:
                    denom = tier if tier >= 1 else 1
                    entity_bonus = entity_score * math.log(1 + entity_occurrences) / denom
                    total_entity_bonus += entity_bonus

            passage_score = self.config.passage_ratio * dpr_passage_score + math.log(1 + total_entity_bonus)

            if apply_attribute_boost:
                overlap = self._attribute_keyword_overlap(question_lower, passage_text_lower)
                if overlap > 0:
                    passage_score += self.config.attribute_keyword_boost * math.log(1 + overlap)

            passage_node_idx = self.node_name_to_vertex_idx[passage_hash_id]
            passage_weights[passage_node_idx] = passage_score * self.config.passage_node_weight
        return passage_weights

    def dense_passage_retrieval(self, question_embedding):
        question_emb = question_embedding.reshape(1, -1)
        question_passage_similarities = np.dot(self.passage_embeddings, question_emb.T).flatten()
        sorted_passage_indices = np.argsort(question_passage_similarities)[::-1]
        sorted_passage_scores = question_passage_similarities[sorted_passage_indices].tolist()
        return sorted_passage_indices, sorted_passage_scores

    def _is_attribute_query(self, question):
        tokens = set(re.findall(r"\w+", question.lower()))
        return any(keyword in tokens for keyword in self.config.attribute_query_keywords)

    def _attribute_keyword_overlap(self, question_lower, passage_text_lower):
        overlap = 0
        for keyword in self.config.attribute_query_keywords:
            if keyword in question_lower and keyword in passage_text_lower:
                overlap += 1
        return overlap
    
    def get_seed_entities(self, question):
        question_entities = list(self.spacy_ner.question_ner(question))
        if len(question_entities) == 0 or len(self.entity_embedding_store.embeddings) == 0:
            return [],[],[],[]
        question_entity_embeddings = self.config.embedding_model.encode(question_entities,normalize_embeddings=True,show_progress_bar=False,batch_size=self.config.batch_size)
        similarities = np.dot(self.entity_embeddings, question_entity_embeddings.T)
        seed_entity_indices = []
        seed_entity_texts = []
        seed_entity_hash_ids = []
        seed_entity_scores = []       
        for query_entity_idx in range(len(question_entities)):
            entity_scores = similarities[:, query_entity_idx]
            best_entity_idx = np.argmax(entity_scores)
            best_entity_score = entity_scores[best_entity_idx]
            best_entity_hash_id = self.entity_hash_ids[best_entity_idx]
            best_entity_text = self.entity_embedding_store.hash_id_to_text[best_entity_hash_id]
            seed_entity_indices.append(best_entity_idx)
            seed_entity_texts.append(best_entity_text)
            seed_entity_hash_ids.append(best_entity_hash_id)
            seed_entity_scores.append(best_entity_score)
        return seed_entity_indices, seed_entity_texts, seed_entity_hash_ids, seed_entity_scores

    def calculate_structured_node_scores(self, question, question_embedding):
        weights = np.zeros(len(self.graph.vs["name"]))
        if not getattr(self.config, "enable_structured_memory_graph", False):
            return weights
        if not self.structured_node_ids:
            return weights
        self._prepare_structured_node_embeddings()
        if self.structured_node_embeddings is None or len(self.structured_node_embeddings) == 0:
            return weights

        question_emb = question_embedding.reshape(1, -1)
        similarities = np.dot(self.structured_node_embeddings, question_emb.T).flatten()
        similarities = self._apply_structured_lexical_boost(question, similarities)
        if len(similarities) == 0:
            return weights

        top_k = min(max(int(getattr(self.config, "structured_node_top_k", 24)), 0), len(self.structured_node_ids))
        if top_k == 0:
            return weights
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        threshold = float(getattr(self.config, "structured_score_threshold", 0.2))
        base_weight = float(getattr(self.config, "structured_node_weight", 0.35))
        intent = self._structured_query_intent(question)

        top_scores = similarities[sorted_indices]
        normalized_scores = min_max_normalize(top_scores.tolist()) if len(top_scores) > 1 else [1.0]
        for local_rank, node_idx in enumerate(sorted_indices):
            score = float(similarities[node_idx])
            if score < threshold and local_rank > 2:
                continue
            node_id = self.structured_node_ids[node_idx]
            vertex_idx = self.node_name_to_vertex_idx.get(node_id)
            if vertex_idx is None:
                continue
            node_type = self.structured_node_types.get(node_id, "")
            type_weight = self._structured_type_weight(node_type)
            status_weight = self._structured_status_weight(node_id, intent)
            weights[vertex_idx] = max(
                weights[vertex_idx],
                base_weight * type_weight * status_weight * max(float(normalized_scores[local_rank]), 0.05),
            )
        return weights

    def _prepare_structured_node_embeddings(self):
        if self.structured_node_embeddings is not None:
            return
        texts = [self.structured_node_texts[node_id] for node_id in self.structured_node_ids]
        if not texts:
            self.structured_node_embeddings = np.array([])
            return
        self.structured_node_embeddings = np.array(
            self.config.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.config.batch_size,
            )
        )

    def _apply_structured_lexical_boost(self, question, similarities):
        boosted = similarities.astype(float).copy()
        question_tokens = set(re.findall(r"\w+", question.lower()))
        time_values = self._query_time_values(question)
        for idx, node_id in enumerate(self.structured_node_ids):
            content = self.structured_node_texts.get(node_id, "").lower()
            node_type = self.structured_node_types.get(node_id, "")
            overlap = sum(1 for token in question_tokens if len(token) > 2 and token in content)
            if overlap:
                boosted[idx] += min(0.2, 0.04 * overlap)
            if node_type == "time" and any(value and value in content for value in time_values):
                boosted[idx] += 0.35
            if node_type == "event" and self._is_temporal_question(question):
                boosted[idx] += 0.05
        return boosted

    @staticmethod
    def _query_time_values(question):
        patterns = [
            r"\b\d{4}-\d{1,2}-\d{1,2}\b",
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
            r"\b(?:19|20)\d{2}\b",
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?\b",
            r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
            r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\b(?:today|yesterday|tomorrow|tonight|last week|next week|last month|next month|last year|next year|recently|currently|now|before|previously)\b",
        ]
        values = []
        for pattern in patterns:
            for match in re.findall(pattern, question, flags=re.IGNORECASE):
                value = str(match).strip().lower()
                if value and value not in values:
                    values.append(value)
        return values

    @staticmethod
    def _structured_query_intent(question):
        lowered = question.lower()
        if any(token in lowered for token in ("current", "currently", "now", "latest", "recent", "recently")):
            return "current"
        if any(token in lowered for token in ("used to", "before", "previously", "earlier", "past", "old")):
            return "history"
        return "neutral"

    @staticmethod
    def _is_temporal_question(question):
        lowered = question.lower()
        return any(token in lowered for token in ("when", "date", "day", "month", "year", "time", "how long", "before", "after", "between"))

    @staticmethod
    def _is_memory_status_question(question):
        lowered = question.lower()
        return any(token in lowered for token in ("current", "currently", "now", "used to", "before", "previously", "prefer", "like", "relation", "who"))

    def _structured_type_weight(self, node_type):
        if node_type == "event":
            return float(getattr(self.config, "structured_event_weight", 1.0))
        if node_type == "time":
            return float(getattr(self.config, "structured_time_weight", 0.8))
        return 1.0

    def _structured_status_weight(self, node_id, intent):
        status = self.structured_node_status.get(node_id, "")
        if not status:
            return 1.0
        if status == "active":
            return 1.1 if intent != "history" else 0.95
        if status == "expired":
            return 1.05 if intent == "history" else 0.65
        if status == "conflicting":
            return 0.75
        return 1.0

    def index(self, passages):
        self.node_to_node_stats = defaultdict(dict)
        self.entity_to_sentence_stats = defaultdict(dict)
        self.passage_embedding_store.insert_text(passages)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()
        existing_passage_hash_id_to_entities,existing_sentence_to_entities, new_passage_hash_ids = self.load_existing_data(hash_id_to_passage.keys())
        if len(new_passage_hash_ids) > 0:
            new_hash_id_to_passage = {k : hash_id_to_passage[k] for k in new_passage_hash_ids}
            new_hash_id_to_ner_text = {
                k: self._visible_passage_text(v) for k, v in new_hash_id_to_passage.items()
            }
            new_passage_hash_id_to_entities,new_sentence_to_entities = self.spacy_ner.batch_ner(new_hash_id_to_ner_text, self.config.max_workers)
            self.merge_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities)
        self.save_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        entity_nodes, sentence_nodes,passage_hash_id_to_entities,self.entity_to_sentence,self.sentence_to_entity = self.extract_nodes_and_edges(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        self.sentence_embedding_store.insert_text(list(sentence_nodes))
        self.entity_embedding_store.insert_text(list(entity_nodes))
        self.entity_hash_id_to_sentence_hash_ids = {}
        for entity, sentence in self.entity_to_sentence.items():
            entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
            self.entity_hash_id_to_sentence_hash_ids[entity_hash_id] = [self.sentence_embedding_store.text_to_hash_id[s] for s in sentence]
        self.sentence_hash_id_to_entity_hash_ids = {}
        for sentence, entities in self.sentence_to_entity.items():
            sentence_hash_id = self.sentence_embedding_store.text_to_hash_id[sentence]
            self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id] = [self.entity_embedding_store.text_to_hash_id[e] for e in entities]
        self.add_entity_to_passage_edges(passage_hash_id_to_entities)
        self.add_adjacent_passage_edges()
        self.add_structured_memory_graph(hash_id_to_passage)
        self.augment_graph()
        output_graphml_path = os.path.join(self.config.working_dir,self.dataset_name, "LinearRAG.graphml")
        os.makedirs(os.path.dirname(output_graphml_path), exist_ok=True)   
        self.graph.write_graphml(output_graphml_path)

    def add_adjacent_passage_edges(self):
        passage_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        index_pattern = re.compile(r'^(\d+):')
        indexed_items = [
            (int(match.group(1)), node_key)
            for node_key, text in passage_id_to_text.items()
            if (match := index_pattern.match(text.strip()))
        ]
        indexed_items.sort(key=lambda x: x[0])
        for i in range(len(indexed_items) - 1):
            current_node = indexed_items[i][1]
            next_node = indexed_items[i + 1][1]
            self.node_to_node_stats[current_node][next_node] = 1.0

    def add_structured_memory_graph(self, hash_id_to_passage):
        self.structured_node_texts = {}
        self.structured_node_types = {}
        self.structured_node_status = {}
        self.structured_node_ids = []
        self.structured_node_embeddings = None
        if not getattr(self.config, "enable_structured_memory_graph", False):
            return

        user_dir = os.path.join(self.config.working_dir, self.dataset_name)
        raw_to_passages = self._raw_id_to_passage_hash_ids(hash_id_to_passage)
        events = self._load_json_list(os.path.join(user_dir, "event_nodes.json"))
        times = self._load_json_list(os.path.join(user_dir, "time_nodes.json"))
        graph_edges = self._load_jsonl(os.path.join(user_dir, "memory_graph_edges.jsonl"))

        valid_time_ids = set()
        for time_node in times:
            time_id = str(time_node.get("time_id") or "")
            if not time_id or self._is_message_timestamp_time(time_node):
                continue
            valid_time_ids.add(time_id)
            self._add_structured_node(time_id, "time", self._time_node_text(time_node))

        for event in events:
            event_id = str(event.get("event_id") or "")
            if not event_id:
                continue
            self._add_structured_node(event_id, "event", self._event_node_text(event))
            raw_id = str(event.get("raw_id") or "")
            for passage_id in raw_to_passages.get(raw_id, []):
                self._add_weighted_edge(event_id, passage_id, 1.0)
            self._connect_structured_node_to_entities(
                event_id,
                [event.get("subject"), event.get("object"), *(event.get("participants") or [])],
                0.45,
            )

        # Consolidated memories are kept as sidecar state for API reranking.
        # The main graph uses only evidence-like nodes to avoid memory-state
        # edges creating broad, noisy activation paths.
        for edge in graph_edges:
            source_id = str(edge.get("source_id") or "")
            target_id = str(edge.get("target_id") or "")
            edge_type = str(edge.get("edge_type") or "")
            if not source_id or not target_id:
                continue
            if edge_type == "evidence":
                if source_id not in self.structured_node_texts:
                    continue
                if self._is_time_structured_node(source_id, valid_time_ids):
                    continue
                for passage_id in raw_to_passages.get(target_id, []):
                    self._add_weighted_edge(source_id, passage_id, 1.0)
            elif edge_type == "occurs_at" and source_id in self.structured_node_texts and target_id in valid_time_ids:
                self._add_weighted_edge(source_id, target_id, 0.85)

        self.structured_node_ids = list(self.structured_node_texts.keys())

    def _add_structured_node(self, node_id, node_type, text):
        if not node_id or not text:
            return
        self.structured_node_texts[node_id] = text
        self.structured_node_types[node_id] = node_type

    def _add_weighted_edge(self, source_id, target_id, weight):
        if not source_id or not target_id or source_id == target_id:
            return
        current = self.node_to_node_stats[source_id].get(target_id, 0.0)
        self.node_to_node_stats[source_id][target_id] = max(float(current), float(weight))

    def _is_time_structured_node(self, node_id, valid_time_ids):
        node_id = str(node_id or "")
        if node_id in valid_time_ids:
            return True
        if self.structured_node_types.get(node_id) == "time":
            return True
        return node_id.startswith("time-")

    @staticmethod
    def _load_json_list(path):
        if not os.path.exists(path):
            return []
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            return []
        return data if isinstance(data, list) else []

    @staticmethod
    def _load_jsonl(path):
        if not os.path.exists(path):
            return []
        records = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
        return records

    @staticmethod
    def _raw_id_to_passage_hash_ids(hash_id_to_passage):
        raw_to_passages = defaultdict(list)
        for passage_hash_id, passage in hash_id_to_passage.items():
            match = re.search(r"\[raw_id=([^\]]+)\]", passage)
            if match:
                raw_to_passages[match.group(1)].append(passage_hash_id)
        return raw_to_passages

    @staticmethod
    def _visible_passage_text(passage):
        text = re.sub(r"^\d+:", "", str(passage), count=1).lstrip()
        lines = text.splitlines()
        if not lines:
            return text.strip()

        first_line = lines[0].strip()
        if re.fullmatch(r"(?:\[[^\]]+\]\s*)+", first_line):
            return "\n".join(lines[1:]).strip()

        text = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", text).lstrip()
        return text.strip()

    @staticmethod
    def _is_message_timestamp_time(time_node):
        expression = str(time_node.get("expression") or "")
        source = str(time_node.get("source") or "")
        kind = str(time_node.get("kind") or "")
        return expression.startswith("message_timestamp:") or source == "message_timestamp" or kind == "message_timestamp"

    @staticmethod
    def _event_node_text(event):
        subject = str(event.get("subject") or "").strip()
        predicate = str(event.get("canonical_trigger") or event.get("event_trigger") or "").strip()
        obj = str(event.get("object") or "").strip()
        source = str(event.get("source_text") or "").strip()
        parts = ["event"]
        if subject or predicate or obj:
            parts.append(f"{subject} {predicate} {obj}".strip())
        if source:
            parts.append(f"source: {source}")
        return ". ".join(part for part in parts if part)

    @staticmethod
    def _memory_node_text(memory):
        memory_type = str(memory.get("memory_type") or "").strip()
        status = str(memory.get("status") or "").strip()
        subject = str(memory.get("subject") or "").strip()
        predicate = str(memory.get("predicate") or "").strip()
        obj = str(memory.get("object") or "").strip()
        source = str(memory.get("source_text") or "").strip()
        parts = [f"memory {memory_type} {status}".strip()]
        if subject or predicate or obj:
            parts.append(f"{subject} {predicate} {obj}".strip())
        if source:
            parts.append(f"source: {source}")
        return ". ".join(part for part in parts if part)

    @staticmethod
    def _time_node_text(time_node):
        expression = str(time_node.get("expression") or "").strip()
        normalized = str(time_node.get("normalized") or "").strip()
        kind = str(time_node.get("kind") or "").strip()
        if normalized and normalized != expression:
            return f"time {normalized}. expression: {expression}. granularity: {kind}"
        return f"time {expression}. granularity: {kind}"

    @staticmethod
    def _memory_evidence_weight(status):
        if status == "active":
            return 1.0
        if status == "expired":
            return 0.35
        if status == "conflicting":
            return 0.55
        return 0.75

    def _connect_structured_node_to_entities(self, node_id, values, weight):
        entity_text_to_hash = {
            self._normalize_for_match(text): hash_id
            for hash_id, text in self.entity_embedding_store.hash_id_to_text.items()
        }
        for value in values:
            norm = self._normalize_for_match(value)
            if not norm or norm == "user":
                continue
            entity_hash_id = entity_text_to_hash.get(norm)
            if entity_hash_id:
                self._add_weighted_edge(node_id, entity_hash_id, weight)

    @staticmethod
    def _normalize_for_match(value):
        text = str(value or "").lower().strip()
        text = re.sub(r"^(?:a|an|the)\s+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip(" \t\r\n\"'`.,!?")

    def augment_graph(self):
        self.add_nodes()
        self.add_edges()

    def add_nodes(self):
        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()} 
        entity_hash_id_to_text = self.entity_embedding_store.get_hash_id_to_text()
        passage_hash_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text, **self.structured_node_texts}
        
        passage_hash_ids = set(passage_hash_id_to_text.keys())
        
        for hash_id, text in all_hash_id_to_text.items():
            if hash_id not in existing_nodes:
                node_type = self.structured_node_types.get(hash_id)
                if node_type:
                    self.graph.add_vertex(name=hash_id, content=text, node_type=node_type)
                else:
                    self.graph.add_vertex(name=hash_id, content=text)
        
        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}   
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[passage_id] 
            for passage_id in passage_hash_ids 
            if passage_id in self.node_name_to_vertex_idx
        ]

    def add_edges(self):
        edges = []
        weights = []
        valid_nodes = set(self.node_name_to_vertex_idx.keys())
        
        for node_hash_id, node_to_node_stats in self.node_to_node_stats.items():
            if node_hash_id not in valid_nodes:
                continue
            for neighbor_hash_id, weight in node_to_node_stats.items():
                if node_hash_id == neighbor_hash_id:
                    continue
                if neighbor_hash_id not in valid_nodes:
                    continue
                edges.append((node_hash_id, neighbor_hash_id))
                weights.append(weight)
        if edges:
            self.graph.add_edges(edges)
            self.graph.es['weight'] = weights

    def add_entity_to_passage_edges(self, passage_hash_id_to_entities):
        passage_to_entity_count ={} 
        passage_to_all_score = defaultdict(int)
        for passage_hash_id, entities in passage_hash_id_to_entities.items():
            passage = self._visible_passage_text(self.passage_embedding_store.hash_id_to_text[passage_hash_id])
            for entity in entities:
                entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
                count = passage.count(entity)
                passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
                passage_to_all_score[passage_hash_id] += count
        for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
            if passage_to_all_score[passage_hash_id] <= 0:
                continue
            score = count / passage_to_all_score[passage_hash_id]
            self.node_to_node_stats[passage_hash_id][entity_hash_id] = score

    def extract_nodes_and_edges(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        entity_nodes = set()
        sentence_nodes = set()
        passage_hash_id_to_entities = defaultdict(set)
        entity_to_sentence= defaultdict(set)
        sentence_to_entity = defaultdict(set)
        for passage_hash_id, entities in existing_passage_hash_id_to_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_hash_id_to_entities[passage_hash_id].add(entity)
        for sentence,entities in existing_sentence_to_entities.items():
            sentence_nodes.add(sentence)
            for entity in entities:
                entity_to_sentence[entity].add(sentence)
                sentence_to_entity[sentence].add(entity)
        return entity_nodes, sentence_nodes, passage_hash_id_to_entities, entity_to_sentence, sentence_to_entity

    def merge_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities):
        existing_passage_hash_id_to_entities.update(new_passage_hash_id_to_entities)
        existing_sentence_to_entities.update(new_sentence_to_entities)
        return existing_passage_hash_id_to_entities, existing_sentence_to_entities

    def save_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        with open(self.ner_results_path, "w") as f:
            json.dump({"passage_hash_id_to_entities": existing_passage_hash_id_to_entities, "sentence_to_entities": existing_sentence_to_entities}, f)
