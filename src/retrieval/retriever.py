import os
import pickle
from typing import Dict, List

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config.settings import AppSettings, setup_global_settings


class HybridRAGRetriever(BaseRetriever):
    """
    Custom Hybrid Retriever that combines:
    1. Vector Search (Semantic) - Weight: 5.0
    2. BM25 Search (Keyword)  - Weight: 3.0
    """

    def __init__(self, top_k: int = 3):
        setup_global_settings()
        self.top_k = top_k

        # --- 1. Setup Vector Retriever (ChromaDB) ---
        self.client = chromadb.HttpClient(
            host=AppSettings.CHROMA_HOST, port=AppSettings.CHROMA_PORT
        )
        self.collection = self.client.get_or_create_collection(
            AppSettings.COLLECTION_NAME
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.vector_index = VectorStoreIndex.from_vector_store(self.vector_store)
        self.vector_retriever = self.vector_index.as_retriever(similarity_top_k=top_k)

        # --- 2. Setup BM25 Retriever (Load Nodes & Rebuild) ---
        print("💾 Loading Pre-Parsed Nodes from disk...")

        if not os.path.exists(AppSettings.NODES_INDEX_PATH):
            raise FileNotFoundError(
                f"❌ Nodes file not found at {AppSettings.NODES_INDEX_PATH}. "
                "Please run 'python src/main.py build-bm25' first."
            )

        with open(AppSettings.NODES_INDEX_PATH, "rb") as f:
            nodes = pickle.load(f)

        # Rebuilding BM25 from nodes is extremely fast (sub-second)
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes, similarity_top_k=top_k
        )
        print(f"✅ BM25 Index Ready ({len(nodes)} nodes loaded).")

    def _normalize_scores(
        self, node_list: List[NodeWithScore]
    ) -> Dict[str, NodeWithScore]:
        """Normalizes scores to 0-1 range to ensure fair weighting."""
        if not node_list:
            return {}
        scores = [n.score for n in node_list]
        min_s, max_s = min(scores), max(scores)

        # Avoid division by zero if all scores are identical
        if max_s == min_s:
            return {n.node.node_id: n for n in node_list}

        normalized_nodes = {}
        for n in node_list:
            # normalize (x - min) / (max - min)
            norm_score = (n.score - min_s) / (max_s - min_s)
            n.score = norm_score  # Update score temporarily for fusion
            normalized_nodes[n.node.node_id] = n
        return normalized_nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str

        # 1. Get results from both retrievers
        vector_nodes = self.vector_retriever.retrieve(query)
        bm25_nodes = self.bm25_retriever.retrieve(query)

        # --- ENHANCED DEBUG PRINTS (Start) ---
        print(f"\n🔍 [DEBUG] Query: '{query}'")
        print(f"   - Vector Candidates: {len(vector_nodes)}")
        if vector_nodes:
            raw_vector_scores = [f"{n.score:.4f}" for n in vector_nodes]
            print(f"     Raw scores: {', '.join(raw_vector_scores)}")

        print(f"   - BM25 Candidates:   {len(bm25_nodes)}")
        if bm25_nodes:
            raw_bm25_scores = [f"{n.score:.4f}" for n in bm25_nodes]
            print(f"     Raw scores: {', '.join(raw_bm25_scores)}")
        # --- DEBUG PRINTS (End) ---

        # Store original scores for debugging before normalization
        original_vector_scores = {n.node.node_id: n.score for n in vector_nodes}
        original_bm25_scores = {n.node.node_id: n.score for n in bm25_nodes}

        # 2. Normalize scores
        norm_vector = self._normalize_scores(vector_nodes)
        norm_bm25 = self._normalize_scores(bm25_nodes)

        # 3. Merge and Apply Formula
        all_ids = set(norm_vector.keys()) | set(norm_bm25.keys())
        final_results = []

        # --- DEBUG: Show component scores for all nodes ---
        if all_ids:
            print(f"\n📊 [DEBUG] Component Scores (normalized 0-1):")
            for node_id in all_ids:
                node_obj = norm_vector.get(node_id) or norm_bm25.get(node_id)
                file_name = (
                    node_obj.metadata.get("file_name", "Unknown")
                    if node_obj
                    else "Unknown"
                )

                # Get scores
                v_raw = original_vector_scores.get(node_id, 0.0)
                b_raw = original_bm25_scores.get(node_id, 0.0)
                v_norm = norm_vector[node_id].score if node_id in norm_vector else 0.0
                b_norm = norm_bm25[node_id].score if node_id in norm_bm25 else 0.0

                # Get weights
                w_v = getattr(AppSettings, "HYBRID_VECTOR_WEIGHT", 5.0)
                w_b = getattr(AppSettings, "HYBRID_BM25_WEIGHT", 3.0)

                # Calculate final score
                final_score = (v_norm * w_v) + (b_norm * w_b)

                print(
                    f"   📄 {file_name[:30]:<30} | "
                    f"Vector: {v_raw:.4f} → {v_norm:.4f} | "
                    f"BM25: {b_raw:.4f} → {b_norm:.4f} | "
                    f"Final: {v_norm:.4f}×{w_v:.1f} + {b_norm:.4f}×{w_b:.1f} = {final_score:.4f}"
                )

        for node_id in all_ids:
            node_obj = norm_vector.get(node_id) or norm_bm25.get(node_id)

            # Get normalized component scores (0.0 to 1.0)
            v_score = norm_vector[node_id].score if node_id in norm_vector else 0.0
            b_score = norm_bm25[node_id].score if node_id in norm_bm25 else 0.0

            # Apply Weights
            w_v = getattr(AppSettings, "HYBRID_VECTOR_WEIGHT", 5.0)
            w_b = getattr(AppSettings, "HYBRID_BM25_WEIGHT", 3.0)

            final_score = (v_score * w_v) + (b_score * w_b)

            node_obj.score = final_score

            # Store component scores in node metadata for later reference
            node_obj.metadata["vector_raw_score"] = original_vector_scores.get(
                node_id, 0.0
            )
            node_obj.metadata["bm25_raw_score"] = original_bm25_scores.get(node_id, 0.0)
            node_obj.metadata["vector_norm_score"] = v_score
            node_obj.metadata["bm25_norm_score"] = b_score
            node_obj.metadata["vector_weight"] = w_v
            node_obj.metadata["bm25_weight"] = w_b

            final_results.append(node_obj)

        # 4. Sort and Return
        final_results.sort(key=lambda x: x.score, reverse=True)
        top_results = final_results[: self.top_k]

        # --- ENHANCED DEBUG PRINTS (Final Results) ---
        print(f"\n🏆 [DEBUG] Top {len(top_results)} Final Results:")
        for i, node in enumerate(top_results, 1):
            file_name = node.metadata.get("file_name", "Unknown")
            v_raw = node.metadata.get("vector_raw_score", 0.0)
            b_raw = node.metadata.get("bm25_raw_score", 0.0)
            v_norm = node.metadata.get("vector_norm_score", 0.0)
            b_norm = node.metadata.get("bm25_norm_score", 0.0)
            w_v = node.metadata.get("vector_weight", 5.0)
            w_b = node.metadata.get("bm25_weight", 3.0)

            print(f"   {i}. {file_name}")
            print(f"      Final Score: {node.score:.4f}")
            print(
                f"      Vector: {v_raw:.4f} → {v_norm:.4f} (×{w_v:.1f} = {v_norm * w_v:.4f})"
            )
            print(
                f"      BM25:   {b_raw:.4f} → {b_norm:.4f} (×{w_b:.1f} = {b_norm * w_b:.4f})"
            )
            print(f"      Preview: {node.text[:80].replace(chr(10), ' ')}...")
            if i < len(top_results):  # Add separator between results
                print(f"      {'─'*60}")
        print("─" * 60)

        return top_results


# Simple test function
if __name__ == "__main__":
    rag = HybridRAGRetriever(top_k=3)
    results = rag.retrieve("What is the deadline for the cohort?")

    print(f"\n🔎 Hybrid Search Results (Top {len(results)}):")
    for i, node in enumerate(results, 1):
        source = node.metadata.get("file_name", "Unknown")
        print(f"\n--- Result {i} (Hybrid Score: {node.score:.4f}) ---")
        print(f"📄 Source: {source}")
        print(f"📝 Text: {node.text[:150]}...")
