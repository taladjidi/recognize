"""Semi-supervised face clustering using two-pass DBSCAN + Agglomerative.

Groups face detections by visual similarity using their 512-dim embeddings.
Runs per-user: each user's faces are clustered independently.

Clustering approach (inspired by Apple Photos):
  Pass 1: DBSCAN with tight threshold — high-precision micro-clusters
  Pass 2: Agglomerative clustering on micro-cluster centroids — merge
          nearby clusters using average linkage for balanced merging

Semi-supervised integration:
  1. Load existing clusters, sample reference detections from each
  2. Gather unclustered ("fresh") and rejected detections
  3. Run two-pass clustering on the combined set
  4. Use voting to map new clusters to existing cluster IDs
  5. Respect per-detection thresholds (negative feedback from user removals)
  6. Create new DB cluster rows for genuinely new clusters

This preserves user work (renames, merges, manual assignments) while still
discovering new clusters from fresh detections.
"""

import logging
import time
import traceback

import numpy as np

log = logging.getLogger(__name__)

# Two-pass clustering parameters for L2-normalized ArcFace 512-dim embeddings.
# Same-person euclidean distance: ~0.4-0.8; different person: ~1.0+
PASS1_EPS = 0.8       # DBSCAN eps — conservative, high-precision micro-clusters
PASS1_MIN_SAMPLES = 2  # Minimum core point density
PASS2_THRESHOLD = 1.0  # Agglomerative merge threshold on centroids (average linkage)
METRIC = "euclidean"

# Voting thresholds (from PHP FaceClusterAnalyzer)
MIN_OVERLAP_EXISTING_CLUSTER = 0.5  # >50% vote overlap → keep existing cluster
MAX_OVERLAP_NEW_CLUSTER = 0.1       # <10% overlap → create new cluster
# Between 10-50% → ambiguous, skip (don't assign)

MIN_DATASET_SIZE = 10  # Minimum detections to attempt clustering
MIN_DETECTION_SIZE = 0.03  # Minimum face bbox size (relative)

# Sampling parameters
MAX_REFERENCE_SAMPLE = 80  # Max detections to sample per existing cluster


def _calculate_centroid(vectors):
    """Calculate the mean centroid of a set of face vectors.

    Args:
        vectors: list of 512-dim vectors (lists or numpy arrays).

    Returns:
        numpy array [512] centroid, or zero vector if empty.
    """
    if len(vectors) == 0:
        return np.zeros(512, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)


def _euclidean_distance(v1, v2):
    """Euclidean distance between two vectors."""
    return float(np.linalg.norm(np.asarray(v1) - np.asarray(v2)))


def _get_reference_sample_size(n_clusters):
    """How many detections to sample per cluster for reference.

    Grows to ~80 for small cluster counts, drops to ~5 for many clusters.
    From PHP: round(75.0 * 2.0^(-0.007 * n) + 5.0)
    """
    return int(round(75.0 * 2.0 ** (-0.007 * n_clusters) + 5.0))


def _two_pass_cluster(embeddings):
    """Two-pass clustering: DBSCAN micro-clusters then agglomerative merge.

    Pass 1 (DBSCAN): Creates small, high-precision clusters with a tight
    distance threshold. False negatives (missed merges) are expected and
    corrected in pass 2.

    Pass 2 (Agglomerative): Computes centroids of pass-1 clusters and merges
    those within PASS2_THRESHOLD using average linkage. This catches same-person
    clusters that were too far apart for DBSCAN's single-point threshold.

    Args:
        embeddings: numpy array [N, 512] of L2-normalized face vectors.

    Returns:
        numpy array [N] of cluster labels (-1 for noise).
    """
    from sklearn.cluster import DBSCAN, AgglomerativeClustering

    # Pass 1: Conservative DBSCAN
    pass1 = DBSCAN(eps=PASS1_EPS, min_samples=PASS1_MIN_SAMPLES,
                   metric=METRIC, n_jobs=-1)
    labels = pass1.fit_predict(embeddings)

    unique_labels = set(labels)
    unique_labels.discard(-1)

    if len(unique_labels) < 2:
        return labels

    # Compute micro-cluster centroids
    centroids = {}
    for lbl in unique_labels:
        mask = labels == lbl
        centroids[lbl] = embeddings[mask].mean(axis=0)

    # Pass 2: Agglomerative merge on centroids
    centroid_labels = list(centroids.keys())
    centroid_arr = np.array([centroids[l] for l in centroid_labels])

    agg = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=PASS2_THRESHOLD,
        metric=METRIC,
        linkage="average",
    )
    merge_labels = agg.fit_predict(centroid_arr)

    # Build merge map and apply
    merge_map = {centroid_labels[i]: int(merge_labels[i])
                 for i in range(len(centroid_labels))}

    final_labels = labels.copy()
    for orig_lbl, new_lbl in merge_map.items():
        if orig_lbl != new_lbl:
            final_labels[labels == orig_lbl] = new_lbl

    n_before = len(unique_labels)
    n_after = len(set(merge_map.values()))
    if n_before != n_after:
        log.info("Two-pass clustering: %d micro-clusters merged to %d clusters",
                 n_before, n_after)

    return final_labels


def cluster_user_faces(db, user_id):
    """Incremental semi-supervised clustering for a single user.

    Preserves existing cluster assignments and user edits. Only assigns
    unclustered (fresh + rejected) detections to clusters.

    Args:
        db: DB instance.
        user_id: Nextcloud user ID string.

    Returns:
        Dict with clustering stats.
    """
    t0 = time.monotonic()

    # Step 1: Load existing clusters and sample reference detections
    existing_clusters = db.get_face_clusters_for_user(user_id)
    sampled_detections = []
    max_votes_by_cluster = {}  # cluster_id → sample count (for vote normalization)

    n_clusters = len(existing_clusters)
    sample_size = _get_reference_sample_size(n_clusters)

    for cluster in existing_clusters:
        cluster_id = cluster["id"]
        sampled = db.get_cluster_sample(cluster_id, sample_size)
        sampled_detections.extend(sampled)
        max_votes_by_cluster[cluster_id] = len(sampled)

    # Step 2: Get unclustered detections (fresh + rejected)
    fresh_detections = db.get_unclustered_detections(
        user_id, min_size=MIN_DETECTION_SIZE,
    )
    rejected_detections = db.get_rejected_detections(
        user_id, min_size=MIN_DETECTION_SIZE,
    )
    unclustered_detections = fresh_detections + rejected_detections
    all_detections = unclustered_detections + sampled_detections

    n_total = len(all_detections)
    n_unclustered = len(unclustered_detections)
    n_fresh = len(fresh_detections)

    if n_total < MIN_DATASET_SIZE or n_fresh == 0:
        log.debug(
            "User %s: not enough data for clustering (%d total, %d fresh)",
            user_id, n_total, n_fresh,
        )
        return {
            "n_faces": n_total, "n_clusters": n_clusters,
            "n_noise": 0, "n_assigned": 0, "elapsed_s": 0.0,
        }

    log.info(
        "User %s: clustering %d detections (%d fresh, %d rejected, %d sampled from %d clusters)",
        user_id, n_total, n_fresh, len(rejected_detections),
        len(sampled_detections), n_clusters,
    )

    # Step 3: Build embedding matrix and run two-pass clustering
    embeddings = np.array(
        [d["face_vector"] for d in all_detections], dtype=np.float32,
    )

    labels = _two_pass_cluster(embeddings)

    # Step 4: Process each cluster — vote to map to existing clusters
    cluster_groups = {}
    for idx, label in enumerate(labels):
        if label < 0:
            continue
        cluster_groups.setdefault(label, []).append(idx)

    n_assigned = 0
    n_new_clusters = 0

    for label, member_indices in cluster_groups.items():
        # Separate unclustered from sampled members
        unclustered_indices = [i for i in member_indices if i < n_unclustered]
        sampled_indices = [i for i in member_indices if i >= n_unclustered]

        if not unclustered_indices:
            continue  # Nothing new to assign

        # Calculate cluster centroid from all members
        cluster_vectors = [all_detections[i]["face_vector"] for i in member_indices]
        centroid = _calculate_centroid(cluster_vectors)

        # Voting: sampled (already-clustered) detections vote for their cluster ID
        votes = {}
        for i in sampled_indices:
            cid = all_detections[i].get("cluster_id")
            if cid is not None and cid > 0:
                votes[cid] = votes.get(cid, 0) + 1

        # Determine target cluster
        target_cluster_id = None
        if votes:
            best_cid = max(votes, key=votes.get)
            if best_cid in max_votes_by_cluster and max_votes_by_cluster[best_cid] > 0:
                overlap = votes[best_cid] / max_votes_by_cluster[best_cid]
            else:
                overlap = 0.0

            if overlap > MIN_OVERLAP_EXISTING_CLUSTER:
                target_cluster_id = best_cid
            elif overlap >= MAX_OVERLAP_NEW_CLUSTER:
                # Ambiguous — skip this cluster
                log.debug("Skipping ambiguous cluster (overlap=%.2f)", overlap)
                continue
        # else: no votes → new cluster

        if target_cluster_id is None:
            # Create a new cluster in the DB
            target_cluster_id = db.create_face_cluster(user_id)
            n_new_clusters += 1

        # Assign unclustered detections, respecting thresholds
        for i in unclustered_indices:
            det = all_detections[i]

            # Threshold check: user removed this face from a cluster before
            threshold = det.get("threshold", 0.0) or 0.0
            if threshold > 0.0:
                dist = _euclidean_distance(centroid, det["face_vector"])
                if dist >= threshold:
                    continue

            db.assign_detection_to_cluster(det["id"], target_cluster_id)
            n_assigned += 1

    # Commit all assignments in one batch
    db.commit()

    elapsed = time.monotonic() - t0
    log.info(
        "User %s: assigned %d detections, %d new clusters, %d existing clusters (%.2fs)",
        user_id, n_assigned, n_new_clusters, n_clusters, elapsed,
    )

    return {
        "n_faces": n_total,
        "n_clusters": n_clusters + n_new_clusters,
        "n_assigned": n_assigned,
        "n_new_clusters": n_new_clusters,
        "n_noise": n_total - n_assigned - len(sampled_detections),
        "elapsed_s": elapsed,
    }


def cluster_all_users(db, user_ids=None, **kwargs):
    """Cluster faces for multiple users.

    Args:
        db: DB instance.
        user_ids: List of user IDs to cluster. If None, discovers all users
                  with face detections from the DB.

    Returns:
        Dict mapping user_id to cluster stats.
    """
    results = {}
    for user_id in user_ids:
        try:
            stats = cluster_user_faces(db, user_id)
            results[user_id] = stats
        except Exception as e:
            log.error(
                "Clustering failed for user %s: %s\n%s",
                user_id, e, traceback.format_exc(),
            )
            results[user_id] = {"error": str(e)}

    return results
