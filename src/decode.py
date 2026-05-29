from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .dna_data import GraphData


def _build_adjacency(graph: GraphData, scores: np.ndarray) -> Dict[int, List[Tuple[int, float]]]:
    adjacency: Dict[int, List[Tuple[int, float]]] = {}
    for edge_id, (src, dst) in enumerate(graph.edge_index.T):
        adjacency.setdefault(int(src), []).append((int(dst), float(scores[edge_id])))
    return adjacency


def _build_adjacency_with_edges(graph: GraphData, scores: np.ndarray) -> Dict[int, List[Tuple[int, int, float]]]:
    adjacency: Dict[int, List[Tuple[int, int, float]]] = {}
    for edge_id, (src, dst) in enumerate(graph.edge_index.T):
        adjacency.setdefault(int(src), []).append((int(dst), edge_id, float(scores[edge_id])))
    return adjacency


def _max_suffix_prefix_overlap(left: str, right: str) -> int:
    max_len = min(len(left), len(right))
    for size in range(max_len, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def _edge_suffix(graph: GraphData, src: int, dst: int) -> str:
    src_text = graph.node_strings[src]
    dst_text = graph.node_strings[dst]
    overlap = _max_suffix_prefix_overlap(src_text, dst_text)
    if overlap <= 0:
        return dst_text[-1:]
    suffix = dst_text[overlap:]
    return suffix if suffix else dst_text[-1:]


def reconstruct_sequence(graph: GraphData, node_path: Sequence[int]) -> str:
    if not node_path:
        return ""
    sequence = graph.node_strings[node_path[0]]
    for src, dst in zip(node_path, node_path[1:]):
        sequence += _edge_suffix(graph, int(src), int(dst))
    return sequence[: graph.target_length]


def greedy_decode(graph: GraphData, scores: np.ndarray) -> str:
    if graph.start_node is None:
        return ""
    adjacency = _build_adjacency(graph, scores)

    path = [graph.start_node]
    current = graph.start_node
    sequence = graph.node_strings[current]
    max_steps = max(graph.target_length * 2, graph.target_length - len(sequence), 1)
    for _ in range(max_steps):
        if len(sequence) >= graph.target_length:
            break
        options = adjacency.get(current, [])
        if not options:
            break
        previous = current
        current = max(options, key=lambda item: item[1])[0]
        sequence += _edge_suffix(graph, previous, current)
        path.append(current)
    return sequence[: graph.target_length]


def beam_search_decode(graph: GraphData, scores: np.ndarray, beam_width: int = 8) -> str:
    candidates = beam_search_candidates(graph, scores, beam_width=beam_width, candidate_limit=1)
    return candidates[0][0] if candidates else ""


def beam_search_candidates(
    graph: GraphData,
    scores: np.ndarray,
    beam_width: int = 8,
    candidate_limit: int = 8,
) -> List[Tuple[str, float]]:
    if graph.start_node is None:
        return []
    adjacency = _build_adjacency(graph, scores)

    start_sequence = graph.node_strings[graph.start_node]
    beam = [([graph.start_node], 0.0, start_sequence)]
    max_steps = max(graph.target_length * 2, graph.target_length - len(start_sequence), 1)
    for _ in range(max_steps):
        if all(len(sequence) >= graph.target_length for _, _, sequence in beam):
            break
        candidates: List[Tuple[List[int], float, str]] = []
        for path, score, sequence in beam:
            if len(sequence) >= graph.target_length:
                candidates.append((path, score, sequence))
                continue
            options = adjacency.get(path[-1], [])
            if not options:
                candidates.append((path, score - 4.0, sequence))
                continue
            for dst, edge_score in options:
                p = min(max(float(edge_score), 1e-6), 1.0 - 1e-6)
                candidates.append(
                    (
                        path + [dst],
                        score + math.log(p),
                        sequence + _edge_suffix(graph, path[-1], dst),
                    )
                )
        if not candidates:
            break
        candidates.sort(key=lambda item: item[1], reverse=True)
        beam = candidates[:beam_width]
    decoded: Dict[str, float] = {}
    for _, score, sequence in beam:
        sequence = sequence[: graph.target_length]
        if sequence and sequence not in decoded:
            decoded[sequence] = score
    return sorted(decoded.items(), key=lambda item: item[1], reverse=True)[:candidate_limit]


def beam_search_candidate_paths(
    graph: GraphData,
    scores: np.ndarray,
    beam_width: int = 16,
    candidate_limit: int = 12,
) -> List[Dict[str, object]]:
    if graph.start_node is None:
        return []
    adjacency = _build_adjacency_with_edges(graph, scores)
    start_sequence = graph.node_strings[graph.start_node]
    beam = [([graph.start_node], [], 0.0, start_sequence)]
    max_steps = max(graph.target_length * 2, graph.target_length - len(start_sequence), 1)
    for _ in range(max_steps):
        if all(len(sequence) >= graph.target_length for _, _, _, sequence in beam):
            break
        candidates: List[Tuple[List[int], List[int], float, str]] = []
        for path, edge_ids, score, sequence in beam:
            if len(sequence) >= graph.target_length:
                candidates.append((path, edge_ids, score, sequence))
                continue
            options = adjacency.get(path[-1], [])
            if not options:
                candidates.append((path, edge_ids, score - 4.0, sequence))
                continue
            for dst, edge_id, edge_score in options:
                p = min(max(float(edge_score), 1e-6), 1.0 - 1e-6)
                candidates.append(
                    (
                        path + [dst],
                        edge_ids + [edge_id],
                        score + math.log(p),
                        sequence + _edge_suffix(graph, path[-1], dst),
                    )
                )
        if not candidates:
            break
        candidates.sort(key=lambda item: item[2], reverse=True)
        beam = candidates[:beam_width]

    deduped: Dict[str, Dict[str, object]] = {}
    for path, edge_ids, score, raw_sequence in beam:
        sequence = raw_sequence[: graph.target_length]
        if not sequence:
            continue
        previous = deduped.get(sequence)
        if previous is None or float(score) > float(previous["score"]):
            deduped[sequence] = {
                "sequence": sequence,
                "raw_sequence": raw_sequence,
                "score": float(score),
                "path": path,
                "edge_ids": edge_ids,
            }
    return sorted(deduped.values(), key=lambda item: float(item["score"]), reverse=True)[:candidate_limit]


def dynamic_lambda(graph: GraphData) -> float:
    edge_counts = graph.edge_counts.astype(np.float32)
    if edge_counts.size == 0:
        return 1.0
    coefficient = float(edge_counts.std() / max(edge_counts.mean(), 1e-6))
    positive_density = float(graph.edge_index.shape[1] / max(len(graph.node_strings), 1))
    coverage_factor = min(max(graph.coverage / 30.0, 0.0), 1.0)

    # 覆盖度低、计数不稳定时更依赖模型；覆盖度高、计数稳定时更依赖丰度。
    lambda_value = 0.65 * (1.0 - coverage_factor) + 0.25 * min(coefficient, 1.0) + 0.10 * min(positive_density / 3.0, 1.0)
    return float(min(max(lambda_value, 0.2), 0.85))


def count_scores(graph: GraphData) -> np.ndarray:
    return graph.edge_counts / max(float(graph.edge_counts.max()), 1.0)


def uncertainty_fused_scores(graph: GraphData, mean_scores: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    uncertainty = np.clip(uncertainty.astype(np.float32), 0.0, 1.0)
    mean_scores = np.clip(mean_scores.astype(np.float32), 1e-6, 1.0 - 1e-6)
    counts = count_scores(graph).astype(np.float32)
    return ((1.0 - uncertainty) * mean_scores + uncertainty * counts).astype(np.float32)


def temperature_scores(scores: np.ndarray, temperature: float) -> np.ndarray:
    scores = np.clip(scores.astype(np.float32), 1e-6, 1.0 - 1e-6)
    logits = np.log(scores / (1.0 - scores))
    scaled = logits / max(float(temperature), 1e-6)
    return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)


def pruned_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    output = scores.astype(np.float32).copy()
    output[output < threshold] = 1e-6
    return output


def fused_scores(graph: GraphData, model_scores: np.ndarray, mode: str) -> np.ndarray:
    counts = count_scores(graph)
    if mode == "model":
        return model_scores
    if mode == "count":
        return counts
    if mode == "fixed":
        lam = 0.5
    elif mode.startswith("lambda_"):
        lam = float(mode.split("_", 1)[1])
    elif mode == "dynamic":
        lam = dynamic_lambda(graph)
    else:
        raise ValueError(f"未知融合模式: {mode}")
    return lam * model_scores + (1.0 - lam) * counts


def read_consistency_score(sequence: str, reads: Sequence[str], k: int) -> float:
    if not sequence or not reads or len(sequence) < k:
        return 0.0
    read_counts: Counter[str] = Counter()
    for read in reads:
        for index in range(0, len(read) - k + 1):
            kmer = read[index : index + k]
            if set(kmer) <= {"A", "C", "G", "T"}:
                read_counts[kmer] += 1
    if not read_counts:
        return 0.0
    max_count = max(read_counts.values())
    normalizer = math.log1p(max_count)
    sequence_kmers = [sequence[index : index + k] for index in range(0, len(sequence) - k + 1)]
    if not sequence_kmers or normalizer <= 0:
        return 0.0
    support = sum(math.log1p(read_counts.get(kmer, 0)) / normalizer for kmer in sequence_kmers)
    return float(support / len(sequence_kmers))


def alignment_consistency_score(sequence: str, reads: Sequence[str]) -> float:
    if not sequence or not reads:
        return 0.0
    scores = []
    for read in reads:
        if not read:
            continue
        distance = edit_distance(sequence, read)
        scores.append(1.0 - distance / max(len(sequence), len(read), 1))
    if not scores:
        return 0.0
    scores.sort(reverse=True)
    keep = max(3, len(scores) // 4)
    return float(np.mean(scores[:keep]))


def local_alignment_polish(
    sequence: str,
    reads: Sequence[str],
    target_length: int,
    max_offset: int = 3,
    min_votes: int = 4,
    majority: float = 0.60,
) -> str:
    if not sequence or not reads:
        return sequence[:target_length]
    sequence = sequence[:target_length]
    votes: List[Counter[str]] = [Counter() for _ in sequence]
    for read in reads:
        if not read:
            continue
        best_offset = 0
        best_matches = -1
        for offset in range(-max_offset, max_offset + 1):
            matches = 0
            compared = 0
            for pos, base in enumerate(sequence):
                read_pos = pos + offset
                if 0 <= read_pos < len(read):
                    compared += 1
                    if read[read_pos] == base:
                        matches += 1
            score = matches - 0.15 * abs(len(sequence) - compared)
            if score > best_matches:
                best_matches = score
                best_offset = offset
        for pos in range(len(sequence)):
            read_pos = pos + best_offset
            if 0 <= read_pos < len(read):
                base = read[read_pos]
                if base in {"A", "C", "G", "T"}:
                    votes[pos][base] += 1

    polished = list(sequence)
    for pos, counter in enumerate(votes):
        total = sum(counter.values())
        if total < min_votes:
            continue
        best_base, best_count = counter.most_common(1)[0]
        current_count = counter.get(polished[pos], 0)
        if (
            best_base != polished[pos]
            and best_count >= max(min_votes, int(np.ceil(total * majority)))
            and best_count >= current_count + 2
        ):
            polished[pos] = best_base
    return "".join(polished)[:target_length]


def indel_edge_risk(graph: GraphData, edge_ids: Sequence[int]) -> float:
    if not edge_ids or graph.edge_features.shape[1] < 13:
        return 0.0
    risks = []
    for edge_id in edge_ids:
        features = graph.edge_features[int(edge_id)]
        support_risk = min(float(features[8] + features[9]), 1.0)
        type_risk = min(float(features[11] + features[12]), 1.0)
        risks.append(0.3 * support_risk + 0.7 * type_risk)
    return float(np.mean(risks)) if risks else 0.0


def sequence_length_penalty(raw_sequence: str, target_length: int) -> float:
    if target_length <= 0:
        return 0.0
    return float(abs(len(raw_sequence) - target_length) / target_length)


def reranked_beam_decode(
    graph: GraphData,
    scores: np.ndarray,
    beam_width: int = 16,
    candidate_limit: int = 16,
    consistency_weight: float = 0.35,
) -> str:
    candidates = beam_search_candidates(graph, scores, beam_width=beam_width, candidate_limit=candidate_limit)
    if not candidates:
        return ""
    raw_scores = np.asarray([score for _, score in candidates], dtype=np.float32)
    if float(raw_scores.max() - raw_scores.min()) < 1e-9:
        model_scores = np.ones_like(raw_scores)
    else:
        model_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    consistency_scores = np.asarray(
        [read_consistency_score(sequence, graph.reads, graph.k) for sequence, _ in candidates],
        dtype=np.float32,
    )
    weight = float(min(max(consistency_weight, 0.0), 1.0))
    final_scores = (1.0 - weight) * model_scores + weight * consistency_scores
    best_index = int(final_scores.argmax())
    return candidates[best_index][0]


def sequence_similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return 1.0 - edit_distance(left, right) / max(len(left), len(right), 1)


def _normalize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    spread = float(array.max() - array.min())
    if spread < 1e-9:
        return np.ones_like(array)
    return (array - array.min()) / spread


def multi_k_consensus_decode(
    graphs: Sequence[GraphData],
    score_sets: Sequence[np.ndarray],
    beam_width: int = 16,
    candidate_limit_per_k: int = 12,
    model_weight: float = 0.40,
    read_weight: float = 0.35,
    agreement_weight: float = 0.25,
) -> Tuple[str, Dict[str, float]]:
    entries: List[Dict[str, float | int | str]] = []
    for graph, scores in zip(graphs, score_sets):
        candidates = beam_search_candidates(
            graph,
            scores,
            beam_width=beam_width,
            candidate_limit=candidate_limit_per_k,
        )
        if not candidates:
            continue
        normalized_path_scores = _normalize([score for _, score in candidates])
        for index, (sequence, raw_score) in enumerate(candidates):
            entries.append(
                {
                    "sequence": sequence,
                    "k": graph.k,
                    "raw_score": float(raw_score),
                    "model_score": float(normalized_path_scores[index]),
                    "read_score": read_consistency_score(sequence, graph.reads, graph.k),
                }
            )

    if not entries:
        return "", {"candidate_count": 0.0, "k_count": 0.0, "agreement_score": 0.0}

    k_values = sorted({int(entry["k"]) for entry in entries})
    for entry in entries:
        sequence = str(entry["sequence"])
        other_k_scores = []
        for k_value in k_values:
            if k_value == int(entry["k"]):
                continue
            other_sequences = [str(other["sequence"]) for other in entries if int(other["k"]) == k_value]
            if other_sequences:
                other_k_scores.append(max(sequence_similarity(sequence, other) for other in other_sequences))
        entry["agreement_score"] = float(np.mean(other_k_scores)) if other_k_scores else 1.0

    total = max(model_weight + read_weight + agreement_weight, 1e-9)
    model_weight /= total
    read_weight /= total
    agreement_weight /= total
    for entry in entries:
        entry["final_score"] = (
            model_weight * float(entry["model_score"])
            + read_weight * float(entry["read_score"])
            + agreement_weight * float(entry["agreement_score"])
        )

    best = max(entries, key=lambda item: float(item["final_score"]))
    return str(best["sequence"]), {
        "candidate_count": float(len(entries)),
        "k_count": float(len(k_values)),
        "best_k": float(best["k"]),
        "model_score": float(best["model_score"]),
        "read_score": float(best["read_score"]),
        "agreement_score": float(best["agreement_score"]),
        "final_score": float(best["final_score"]),
    }


def multi_k_indel_consensus_decode(
    graphs: Sequence[GraphData],
    score_sets: Sequence[np.ndarray],
    beam_width: int = 16,
    candidate_limit_per_k: int = 12,
    model_weight: float = 0.35,
    read_weight: float = 0.30,
    agreement_weight: float = 0.15,
    alignment_weight: float = 0.15,
    risk_weight: float = 0.10,
    length_weight: float = 0.05,
    risk_filter: float = 1.0,
    min_read_score: float = 0.0,
    max_length_penalty: float = 1.0,
    primary_k: int | None = None,
    protect_single_k: bool = False,
    protection_margin: float = 0.06,
    polish: bool = False,
) -> Tuple[str, Dict[str, float]]:
    entries: List[Dict[str, float | int | str]] = []
    for graph, scores in zip(graphs, score_sets):
        candidates = beam_search_candidate_paths(
            graph,
            scores,
            beam_width=beam_width,
            candidate_limit=candidate_limit_per_k,
        )
        if not candidates:
            continue
        normalized_path_scores = _normalize([float(item["score"]) for item in candidates])
        for index, candidate in enumerate(candidates):
            sequence = str(candidate["sequence"])
            raw_sequence = str(candidate["raw_sequence"])
            if polish:
                polished = local_alignment_polish(sequence, graph.reads, graph.target_length)
                original_alignment = alignment_consistency_score(sequence, graph.reads)
                polished_alignment = alignment_consistency_score(polished, graph.reads)
                original_read = read_consistency_score(sequence, graph.reads, graph.k)
                polished_read = read_consistency_score(polished, graph.reads, graph.k)
                if polished_alignment >= original_alignment + 0.003 and polished_read >= original_read - 0.005:
                    sequence = polished
            edge_ids = [int(edge_id) for edge_id in candidate["edge_ids"]]
            entries.append(
                {
                    "sequence": sequence,
                    "k": graph.k,
                    "raw_score": float(candidate["score"]),
                    "model_score": float(normalized_path_scores[index]),
                    "read_score": read_consistency_score(sequence, graph.reads, graph.k),
                    "alignment_score": alignment_consistency_score(sequence, graph.reads),
                    "edge_risk": indel_edge_risk(graph, edge_ids),
                    "length_penalty": sequence_length_penalty(raw_sequence, graph.target_length),
                }
            )

    if not entries:
        return "", {
            "candidate_count": 0.0,
            "k_count": 0.0,
            "agreement_score": 0.0,
            "alignment_score": 0.0,
            "edge_risk": 0.0,
            "length_penalty": 0.0,
        }

    k_values = sorted({int(entry["k"]) for entry in entries})
    for entry in entries:
        sequence = str(entry["sequence"])
        other_k_scores = []
        for k_value in k_values:
            if k_value == int(entry["k"]):
                continue
            other_sequences = [str(other["sequence"]) for other in entries if int(other["k"]) == k_value]
            if other_sequences:
                other_k_scores.append(max(sequence_similarity(sequence, other) for other in other_sequences))
        entry["agreement_score"] = float(np.mean(other_k_scores)) if other_k_scores else 1.0

    for entry in entries:
        entry["final_score"] = (
            model_weight * float(entry["model_score"])
            + read_weight * float(entry["read_score"])
            + agreement_weight * float(entry["agreement_score"])
            + alignment_weight * float(entry["alignment_score"])
            - risk_weight * float(entry["edge_risk"])
            - length_weight * float(entry["length_penalty"])
        )

    filtered_entries = [
        entry
        for entry in entries
        if float(entry["edge_risk"]) <= risk_filter
        and float(entry["read_score"]) >= min_read_score
        and float(entry["length_penalty"]) <= max_length_penalty
    ]
    selected_entries = filtered_entries if filtered_entries else entries
    best = max(selected_entries, key=lambda item: float(item["final_score"]))
    protected = None
    if protect_single_k:
        target_k = primary_k
        if target_k is None and graphs:
            sorted_k = sorted(graph.k for graph in graphs)
            target_k = sorted_k[len(sorted_k) // 2]
        protected_candidates = [
            entry
            for entry in entries
            if int(entry["k"]) == int(target_k)
            and float(entry["edge_risk"]) <= min(risk_filter + 0.10, 1.0)
            and float(entry["length_penalty"]) <= max_length_penalty
        ]
        if protected_candidates:
            protected = max(
                protected_candidates,
                key=lambda item: (float(item["final_score"]), float(item["model_score"])),
            )
            best_advantage = float(best["final_score"]) - float(protected["final_score"])
            best_risk_gap = float(best["edge_risk"]) - float(protected["edge_risk"])
            if best_advantage <= protection_margin or best_risk_gap >= 0.10:
                best = protected
    return str(best["sequence"]), {
        "candidate_count": float(len(entries)),
        "filtered_candidate_count": float(len(filtered_entries)),
        "k_count": float(len(k_values)),
        "best_k": float(best["k"]),
        "model_score": float(best["model_score"]),
        "read_score": float(best["read_score"]),
        "agreement_score": float(best["agreement_score"]),
        "alignment_score": float(best["alignment_score"]),
        "edge_risk": float(best["edge_risk"]),
        "length_penalty": float(best["length_penalty"]),
        "final_score": float(best["final_score"]),
        "used_single_k_protection": 1.0 if protected is not None and best is protected else 0.0,
    }


def edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]
