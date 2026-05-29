from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np


DNA_ALPHABET = "ACGT"
BASE_TO_INDEX = {base: index for index, base in enumerate(DNA_ALPHABET)}


@dataclass(frozen=True)
class ChannelConfig:
    sequence_length: int = 100
    coverage: int = 20
    substitution_rate: float = 0.018
    insertion_rate: float = 0.012
    deletion_rate: float = 0.018
    contaminant_rate: float = 0.04


@dataclass
class ReadCluster:
    cluster_id: int
    reference: str
    reads: List[str]
    coverage: int
    read_is_contaminant: List[int] = field(default_factory=list)
    read_weights: List[float] = field(default_factory=list)


@dataclass
class GraphData:
    cluster_id: int
    k: int
    coverage: int
    reference: str
    node_strings: List[str]
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    edge_labels: np.ndarray
    edge_counts: np.ndarray
    start_node: int | None
    target_length: int
    reads: List[str] = field(default_factory=list)
    read_weights: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float32))
    read_features: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    read_labels: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float32))
    contamination_rate: float = 0.0
    weighted_graph: bool = False


def random_dna(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(DNA_ALPHABET) for _ in range(length))


def mutate_sequence(reference: str, rng: random.Random, config: ChannelConfig) -> str:
    read: List[str] = []
    for base in reference:
        if rng.random() < config.deletion_rate:
            continue
        if rng.random() < config.substitution_rate:
            candidates = [item for item in DNA_ALPHABET if item != base]
            read.append(rng.choice(candidates))
        else:
            read.append(base)
        if rng.random() < config.insertion_rate:
            read.append(rng.choice(DNA_ALPHABET))
    return "".join(read)


def generate_clusters(num_clusters: int, config: ChannelConfig, seed: int) -> List[ReadCluster]:
    rng = random.Random(seed)
    clusters: List[ReadCluster] = []
    contaminants = round(config.coverage * config.contaminant_rate)
    real_reads = max(config.coverage - contaminants, 1)
    for cluster_id in range(num_clusters):
        reference = random_dna(config.sequence_length, rng)
        labeled_reads = [(mutate_sequence(reference, rng, config), 0) for _ in range(real_reads)]
        for _ in range(contaminants):
            contaminant_length = max(20, config.sequence_length + rng.randint(-12, 12))
            labeled_reads.append((random_dna(contaminant_length, rng), 1))
        rng.shuffle(labeled_reads)
        reads = [read for read, _ in labeled_reads]
        read_labels = [label for _, label in labeled_reads]
        clusters.append(
            ReadCluster(
                cluster_id=cluster_id,
                reference=reference,
                reads=reads,
                coverage=config.coverage,
                read_is_contaminant=read_labels,
            )
        )
    return clusters


def _is_cnr_separator(line: str) -> bool:
    return bool(line) and set(line) == {"="}


def _clean_dna_line(line: str) -> str:
    return "".join(base for base in line.strip().upper() if base in DNA_ALPHABET)


def _iter_cnr_blocks(clusters_path: Path) -> Iterator[List[str]]:
    current: List[str] | None = None
    with clusters_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            if _is_cnr_separator(line):
                if current is not None:
                    yield current
                current = []
                continue
            if current is None:
                current = []
            read = _clean_dna_line(line)
            if read:
                current.append(read)
    if current is not None:
        yield current


def load_cnr_clusters(
    raw_dir: Path,
    max_clusters: int | None = None,
    min_reads: int = 30,
    max_reads_per_cluster: int | None = None,
    seed: int = 42,
) -> List[ReadCluster]:
    centers_path = raw_dir / "Centers.txt"
    clusters_path = raw_dir / "Clusters.txt"
    if not centers_path.exists() or not clusters_path.exists():
        raise FileNotFoundError(
            f"需要在 {raw_dir} 下找到 Centers.txt 和 Clusters.txt。"
            "请先获取 https://github.com/microsoft/clustered-nanopore-reads-dataset 数据。"
        )

    rng = random.Random(seed)
    centers = [_clean_dna_line(line) for line in centers_path.read_text(encoding="utf-8").splitlines()]
    selected: List[ReadCluster] = []
    for cluster_id, (reference, reads) in enumerate(zip(centers, _iter_cnr_blocks(clusters_path))):
        if not reference or len(reads) < min_reads:
            continue
        sampled_reads = list(reads)
        if max_reads_per_cluster is not None and len(sampled_reads) > max_reads_per_cluster:
            sampled_reads = rng.sample(sampled_reads, max_reads_per_cluster)
        selected.append(
            ReadCluster(
                cluster_id=cluster_id,
                reference=reference,
                reads=sampled_reads,
                coverage=len(sampled_reads),
                read_is_contaminant=[0] * len(sampled_reads),
            )
        )
        if max_clusters is not None and len(selected) >= max_clusters:
            break
    return selected


def _cluster_read_labels(cluster: ReadCluster) -> List[int]:
    if len(cluster.read_is_contaminant) == len(cluster.reads):
        return [1 if int(label) else 0 for label in cluster.read_is_contaminant]
    return [0] * len(cluster.reads)


def _sample_external_read(
    source_cluster_id: int,
    clusters: Sequence[ReadCluster],
    rng: random.Random,
    fallback_length: int,
) -> str:
    candidates = [cluster for cluster in clusters if cluster.cluster_id != source_cluster_id and cluster.reads]
    if not candidates:
        return random_dna(max(fallback_length, 20), rng)
    donor = rng.choice(candidates)
    return rng.choice(donor.reads)


def contaminate_clusters(
    clusters: Sequence[ReadCluster],
    contamination_rate: float,
    seed: int,
    mode: str = "replace",
    preserve_ids: bool = True,
    cluster_id_offset: int = 0,
) -> List[ReadCluster]:
    if mode not in {"replace", "append"}:
        raise ValueError("mode 必须是 'replace' 或 'append'。")
    rate = float(min(max(contamination_rate, 0.0), 1.0))
    rng = random.Random(seed)
    contaminated: List[ReadCluster] = []
    for output_index, cluster in enumerate(clusters):
        reads = list(cluster.reads)
        labels = _cluster_read_labels(cluster)
        if not reads:
            continue
        contam_count = int(round(len(reads) * rate))
        if rate > 0.0:
            contam_count = max(contam_count, 1)
        fallback_length = int(np.median([len(read) for read in reads])) if reads else len(cluster.reference)
        contaminant_reads = [
            _sample_external_read(cluster.cluster_id, clusters, rng, fallback_length) for _ in range(contam_count)
        ]
        if mode == "append":
            reads.extend(contaminant_reads)
            labels.extend([1] * len(contaminant_reads))
        elif contaminant_reads:
            positions = rng.sample(range(len(reads)), min(len(contaminant_reads), len(reads)))
            for position, contaminant_read in zip(positions, contaminant_reads):
                reads[position] = contaminant_read
                labels[position] = 1
        new_cluster_id = cluster.cluster_id if preserve_ids else cluster_id_offset + output_index
        contaminated.append(
            ReadCluster(
                cluster_id=new_cluster_id,
                reference=cluster.reference,
                reads=reads,
                coverage=len(reads),
                read_is_contaminant=labels,
            )
        )
    return contaminated


def iter_kmers(sequence: str, k: int) -> Iterable[str]:
    for index in range(0, len(sequence) - k + 1):
        yield sequence[index : index + k]


def _edit_distance(left: str, right: str) -> int:
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


def _preliminary_beam_sequence(
    cluster: ReadCluster,
    k: int,
    threshold: int | None,
    indel_aware: bool,
    max_indel_shift: int,
) -> str:
    try:
        from .decode import beam_search_decode, count_scores

        graph = build_debruijn_graph(
            cluster,
            k,
            threshold=threshold,
            indel_aware=indel_aware,
            max_indel_shift=max_indel_shift,
            read_weighting=False,
            attach_read_features=False,
        )
        if graph is not None:
            sequence = beam_search_decode(graph, count_scores(graph), beam_width=8)
            if sequence:
                return sequence
    except Exception:
        pass
    return max(cluster.reads, key=len, default="")


def estimate_read_weights(
    cluster: ReadCluster,
    k: int,
    threshold: int | None = None,
    indel_aware: bool = False,
    max_indel_shift: int = 3,
    min_weight: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    kmer_counts: Dict[str, int] = {}
    for read in cluster.reads:
        for kmer in iter_kmers(read, k):
            if set(kmer) <= set(DNA_ALPHABET):
                kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
    max_count = max(kmer_counts.values(), default=1)
    log_normalizer = np.log1p(max_count)

    lengths = np.asarray([len(read) for read in cluster.reads], dtype=np.float32)
    median_length = float(np.median(lengths)) if lengths.size else float(len(cluster.reference))
    gcs = np.asarray([gc_fraction(read) for read in cluster.reads], dtype=np.float32)
    median_gc = float(np.median(gcs)) if gcs.size else gc_fraction(cluster.reference)
    preliminary = _preliminary_beam_sequence(cluster, k, threshold, indel_aware, max_indel_shift)

    weights = []
    features = []
    for read in cluster.reads:
        read_kmers = [kmer for kmer in iter_kmers(read, k) if set(kmer) <= set(DNA_ALPHABET)]
        if read_kmers and log_normalizer > 0:
            kmer_score = float(np.mean([np.log1p(kmer_counts.get(kmer, 0)) / log_normalizer for kmer in read_kmers]))
        else:
            kmer_score = 0.0

        length_score = 1.0 - min(abs(len(read) - median_length) / max(median_length, 1.0), 1.0)
        gc_score = 1.0 - min(abs(gc_fraction(read) - median_gc) / 0.5, 1.0)
        if preliminary:
            distance = _edit_distance(read, preliminary)
            edit_score = 1.0 - distance / max(len(read), len(preliminary), 1)
        else:
            edit_score = 0.0

        weight = 0.45 * kmer_score + 0.20 * length_score + 0.35 * edit_score
        weight = float(min(max(weight, min_weight), 1.0))
        weights.append(weight)
        features.append([kmer_score, length_score, edit_score, gc_score, weight])

    labels = np.asarray(_cluster_read_labels(cluster), dtype=np.float32)
    return (
        np.asarray(weights, dtype=np.float32),
        np.asarray(features, dtype=np.float32),
        labels,
    )


def one_hot_kmer(kmer: str) -> np.ndarray:
    encoded = np.zeros((len(kmer), 4), dtype=np.float32)
    for index, base in enumerate(kmer):
        if base in BASE_TO_INDEX:
            encoded[index, BASE_TO_INDEX[base]] = 1.0
    return encoded.reshape(-1)


def gc_fraction(sequence: str) -> float:
    if not sequence:
        return 0.0
    return sum(base in "GC" for base in sequence) / len(sequence)


def abundance_threshold_for_coverage(coverage: int) -> int:
    if coverage <= 8:
        return 1
    if coverage <= 15:
        return 2
    return 2


def _max_suffix_prefix_overlap(left: str, right: str) -> int:
    max_len = min(len(left), len(right))
    for size in range(max_len, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def build_debruijn_graph(
    cluster: ReadCluster,
    k: int,
    threshold: int | None = None,
    indel_aware: bool = False,
    max_indel_shift: int = 3,
    read_weighting: bool = False,
    attach_read_features: bool = False,
) -> GraphData | None:
    threshold = abundance_threshold_for_coverage(cluster.coverage) if threshold is None else threshold
    read_weights = np.ones(len(cluster.reads), dtype=np.float32)
    read_features = np.zeros((0, 0), dtype=np.float32)
    read_labels = np.asarray(_cluster_read_labels(cluster), dtype=np.float32)
    if read_weighting or attach_read_features:
        read_weights, read_features, read_labels = estimate_read_weights(
            cluster,
            k,
            threshold=threshold,
            indel_aware=indel_aware,
            max_indel_shift=max_indel_shift,
        )
    elif len(cluster.read_weights) == len(cluster.reads):
        read_weights = np.asarray(cluster.read_weights, dtype=np.float32)

    counts: Dict[str, float] = {}
    for read_index, read in enumerate(cluster.reads):
        read_weight = float(read_weights[read_index]) if read_weighting and read_index < len(read_weights) else 1.0
        for kmer in iter_kmers(read, k):
            if set(kmer) <= set(DNA_ALPHABET):
                counts[kmer] = counts.get(kmer, 0.0) + read_weight

    kept = {kmer: count for kmer, count in counts.items() if count >= threshold}
    if not kept:
        return None

    node_to_id: Dict[str, int] = {}
    edge_support: Dict[Tuple[int, int], np.ndarray] = {}

    def node_id(value: str) -> int:
        if value not in node_to_id:
            node_to_id[value] = len(node_to_id)
        return node_to_id[value]

    def add_edge(src: int, dst: int, support: float, channel: int) -> None:
        if support <= 0:
            return
        key = (src, dst)
        if key not in edge_support:
            edge_support[key] = np.zeros(3, dtype=np.float32)
        edge_support[key][channel] += float(support)

    for kmer, count in kept.items():
        src = node_id(kmer[:-1])
        dst = node_id(kmer[1:])
        add_edge(src, dst, count, 0)

    if not edge_support:
        return None

    node_strings = [""] * len(node_to_id)
    for node, idx in node_to_id.items():
        node_strings[idx] = node

    node_length = k - 1
    if indel_aware:
        bridge_counts: Dict[Tuple[int, int], float] = {}
        bypass_counts: Dict[Tuple[int, int], float] = {}
        for read_index, read in enumerate(cluster.reads):
            read_weight = float(read_weights[read_index]) if read_weighting and read_index < len(read_weights) else 1.0
            for shift in range(2, max(max_indel_shift, 2) + 1):
                if len(read) < node_length + shift:
                    continue
                for index in range(0, len(read) - node_length - shift + 1):
                    src = node_to_id.get(read[index : index + node_length])
                    dst = node_to_id.get(read[index + shift : index + shift + node_length])
                    if src is not None and dst is not None:
                        bridge_counts[(src, dst)] = bridge_counts.get((src, dst), 0.0) + read_weight

            window_length = k + 1
            if len(read) >= window_length:
                for index in range(0, len(read) - window_length + 1):
                    window = read[index : index + window_length]
                    for drop_index in range(1, window_length - 1):
                        candidate = window[:drop_index] + window[drop_index + 1 :]
                        src = node_to_id.get(candidate[:-1])
                        dst = node_to_id.get(candidate[1:])
                        if src is not None and dst is not None:
                            bypass_counts[(src, dst)] = bypass_counts.get((src, dst), 0.0) + read_weight

        min_support = max(1, threshold)
        for (src, dst), count in bridge_counts.items():
            overlap = _max_suffix_prefix_overlap(node_strings[src], node_strings[dst])
            shift = node_length - overlap
            if count >= min_support and 2 <= shift <= max(max_indel_shift, 2):
                add_edge(src, dst, count, 1)
        for (src, dst), count in bypass_counts.items():
            if count >= min_support:
                add_edge(src, dst, count, 2)

    edge_pairs = list(edge_support.keys())
    edge_index = np.asarray(edge_pairs, dtype=np.int64).T
    support_matrix = np.stack([edge_support[pair] for pair in edge_pairs], axis=0).astype(np.float32)
    edge_counts = support_matrix.sum(axis=1).astype(np.float32)
    num_nodes = len(node_strings)

    in_degree = np.zeros(num_nodes, dtype=np.float32)
    out_degree = np.zeros(num_nodes, dtype=np.float32)
    abundance = np.zeros(num_nodes, dtype=np.float32)
    for (src, dst), count in zip(edge_pairs, edge_counts):
        out_degree[src] += 1.0
        in_degree[dst] += 1.0
        abundance[src] += count
        abundance[dst] += count

    max_in = max(float(in_degree.max()), 1.0)
    max_out = max(float(out_degree.max()), 1.0)
    max_abundance = max(float(abundance.max()), 1.0)
    max_log_abundance = np.log1p(max_abundance)

    node_features = []
    for idx, node in enumerate(node_strings):
        stats = np.asarray(
            [
                in_degree[idx] / max_in,
                out_degree[idx] / max_out,
                abundance[idx] / max_abundance,
                np.log1p(abundance[idx]) / max_log_abundance,
                gc_fraction(node),
            ],
            dtype=np.float32,
        )
        node_features.append(np.concatenate([stats, one_hot_kmer(node)]))
    node_features_array = np.stack(node_features, axis=0).astype(np.float32)

    max_edge = max(float(edge_counts.max()), 1.0)
    max_log_edge = np.log1p(max_edge)
    edge_features = []
    max_normal = max(float(support_matrix[:, 0].max()), 1.0)
    max_bridge = max(float(support_matrix[:, 1].max()), 1.0)
    max_bypass = max(float(support_matrix[:, 2].max()), 1.0)
    max_shift = max(float(max_indel_shift), 1.0)
    for edge_row, ((src, dst), count) in enumerate(zip(edge_pairs, edge_counts)):
        abundance_similarity = 1.0 - abs(abundance[src] - abundance[dst]) / max_abundance
        features = [
            count / max_edge,
            np.log1p(count) / max_log_edge,
            abundance_similarity,
            in_degree[dst] / max_in,
            out_degree[src] / max_out,
        ]
        if indel_aware:
            overlap = _max_suffix_prefix_overlap(node_strings[src], node_strings[dst])
            shift = max(node_length - overlap, 1)
            dominant_channel = int(np.argmax(support_matrix[edge_row]))
            type_features = [1.0 if dominant_channel == index else 0.0 for index in range(3)]
            features.extend(
                [
                    overlap / max(float(node_length), 1.0),
                    min(float(shift) / max_shift, 1.0),
                    support_matrix[edge_row, 0] / max_normal,
                    support_matrix[edge_row, 1] / max_bridge,
                    support_matrix[edge_row, 2] / max_bypass,
                    *type_features,
                ]
            )
        edge_features.append(features)
    edge_features_array = np.asarray(edge_features, dtype=np.float32)

    true_edges = set()
    max_true_shift = max(max_indel_shift if indel_aware else 1, 1)
    for index in range(0, len(cluster.reference) - node_length + 1):
        src = cluster.reference[index : index + node_length]
        if src not in node_to_id:
            continue
        for shift in range(1, max_true_shift + 1):
            dst_start = index + shift
            dst_end = dst_start + node_length
            if dst_end > len(cluster.reference):
                continue
            dst = cluster.reference[dst_start:dst_end]
            if dst in node_to_id:
                true_edges.add((node_to_id[src], node_to_id[dst]))
    edge_labels = np.asarray([1.0 if pair in true_edges else 0.0 for pair in edge_pairs], dtype=np.float32)

    start_node = node_to_id.get(cluster.reference[: k - 1])
    return GraphData(
        cluster_id=cluster.cluster_id,
        k=k,
        coverage=cluster.coverage,
        reference=cluster.reference,
        node_strings=node_strings,
        node_features=node_features_array,
        edge_index=edge_index,
        edge_features=edge_features_array,
        edge_labels=edge_labels,
        edge_counts=edge_counts,
        start_node=start_node,
        target_length=len(cluster.reference),
        reads=list(cluster.reads),
        read_weights=read_weights.astype(np.float32),
        read_features=read_features.astype(np.float32),
        read_labels=read_labels.astype(np.float32),
        contamination_rate=float(read_labels.mean()) if read_labels.size else 0.0,
        weighted_graph=read_weighting,
    )


def pad_node_feature_dims(graphs: List[GraphData]) -> List[GraphData]:
    if not graphs:
        return graphs
    max_node_dim = max(graph.node_features.shape[1] for graph in graphs)
    for graph in graphs:
        pad_width = max_node_dim - graph.node_features.shape[1]
        if pad_width > 0:
            graph.node_features = np.pad(graph.node_features, ((0, 0), (0, pad_width)), mode="constant")
    return graphs


def graphs_from_clusters(
    clusters: List[ReadCluster],
    k_values: List[int],
    threshold: int | None = None,
    require_start_node: bool = True,
    indel_aware: bool = False,
    max_indel_shift: int = 3,
    read_weighting: bool = False,
    attach_read_features: bool = False,
) -> List[GraphData]:
    all_graphs: List[GraphData] = []
    for cluster in clusters:
        for k in k_values:
            graph = build_debruijn_graph(
                cluster,
                k,
                threshold=threshold,
                indel_aware=indel_aware,
                max_indel_shift=max_indel_shift,
                read_weighting=read_weighting,
                attach_read_features=attach_read_features,
            )
            if graph is None:
                continue
            if require_start_node and graph.start_node is None:
                continue
            if graph.edge_labels.sum() <= 0:
                continue
            all_graphs.append(graph)
    return pad_node_feature_dims(all_graphs)


def make_graph_dataset(
    num_clusters: int,
    coverage_values: List[int],
    k_values: List[int],
    seed: int,
    sequence_length: int = 100,
    substitution_rate: float = 0.018,
    insertion_rate: float = 0.012,
    deletion_rate: float = 0.018,
    contaminant_rate: float = 0.04,
    indel_aware: bool = False,
    max_indel_shift: int = 3,
    read_weighting: bool = False,
    attach_read_features: bool = False,
) -> List[GraphData]:
    all_graphs: List[GraphData] = []
    cluster_offset = 0
    for coverage_index, coverage in enumerate(coverage_values):
        config = ChannelConfig(
            sequence_length=sequence_length,
            coverage=coverage,
            substitution_rate=substitution_rate,
            insertion_rate=insertion_rate,
            deletion_rate=deletion_rate,
            contaminant_rate=contaminant_rate,
        )
        clusters = generate_clusters(
            num_clusters=num_clusters,
            config=config,
            seed=seed + coverage_index * 1009,
        )
        for cluster in clusters:
            cluster.cluster_id += cluster_offset
            for k in k_values:
                graph = build_debruijn_graph(
                    cluster,
                    k,
                    indel_aware=indel_aware,
                    max_indel_shift=max_indel_shift,
                    read_weighting=read_weighting,
                    attach_read_features=attach_read_features,
                )
                if graph is not None and graph.start_node is not None and graph.edge_labels.sum() > 0:
                    all_graphs.append(graph)
        cluster_offset += num_clusters
    return pad_node_feature_dims(all_graphs)
