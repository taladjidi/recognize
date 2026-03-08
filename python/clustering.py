"""Face clustering using HDBSCAN.

Groups face detections by visual similarity using their 512-dim embeddings.
Runs per-user: each user's faces are clustered independently.

Algorithm:
  1. Fetch all face detections for a user from DB
  2. Run HDBSCAN on the 512-dim L2-normalized embeddings
  3. Write cluster IDs back to DB (-1 noise points become NULL)

HDBSCAN parameters are tuned for face recognition:
  - min_cluster_size=3: At least 3 photos to form a "person"
  - min_samples=2: Allow small clusters (pairs of photos)
  - metric="euclidean": Works on L2-normalized embeddings (euclidean distance is
    monotonically related to cosine distance for unit vectors: d²=2(1-cos_sim))
  - cluster_selection_epsilon=1.0: Merge nearby clusters — on unit vectors,
    d_euc=1.0 corresponds to cosine_sim=0.5, suitable for same-person variations
  - allow_single_cluster=True: Required so a dataset with only one person still
    forms a cluster instead of being labeled as all-noise
"""

import logging
import time

import numpy as np

log = logging.getLogger(__name__)

# HDBSCAN tuning parameters
MIN_CLUSTER_SIZE = 3
MIN_SAMPLES = 2
METRIC = "euclidean"
CLUSTER_SELECTION_EPSILON = 1.0
ALLOW_SINGLE_CLUSTER = True


def cluster_faces(embeddings, min_cluster_size=MIN_CLUSTER_SIZE,
                  min_samples=MIN_SAMPLES, metric=METRIC,
                  cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
                  allow_single_cluster=ALLOW_SINGLE_CLUSTER):
    """Cluster face embeddings using HDBSCAN.

    Args:
        embeddings: numpy array [N, 512] of L2-normalized face vectors.
        min_cluster_size: Minimum faces to form a cluster (person).
        min_samples: Core point density threshold.
        metric: Distance metric.
        cluster_selection_epsilon: Merge threshold for nearby clusters.
        allow_single_cluster: If True, allows forming a single cluster.

    Returns:
        numpy array of cluster labels, length N.
        -1 means noise (unclustered).
    """
    from sklearn.cluster import HDBSCAN

    if len(embeddings) < min_cluster_size:
        return np.full(len(embeddings), -1, dtype=np.intp)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
        allow_single_cluster=allow_single_cluster,
        store_centers="centroid",
    )
    labels = clusterer.fit_predict(embeddings)
    return labels


def cluster_user_faces(db, user_id, **kwargs):
    """Cluster all faces for a single user and write results to DB.

    Args:
        db: DB instance.
        user_id: Nextcloud user ID string.
        **kwargs: Override HDBSCAN parameters.

    Returns:
        Dict with clustering stats: {n_faces, n_clusters, n_noise, elapsed_s}.
    """
    t0 = time.monotonic()

    faces = db.get_face_detections_for_user(user_id)
    n_faces = len(faces)

    if n_faces == 0:
        return {"n_faces": 0, "n_clusters": 0, "n_noise": 0, "elapsed_s": 0.0}

    # Extract embeddings and IDs
    ids = [f["id"] for f in faces]
    embeddings = np.array([f["face_vector"] for f in faces], dtype=np.float32)

    # Run clustering
    labels = cluster_faces(embeddings, **kwargs)

    # Write cluster IDs back to DB
    db.update_face_clusters(ids, labels.tolist())

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    elapsed = time.monotonic() - t0

    log.info(
        "Clustered %d faces for user %s: %d clusters, %d noise (%.2fs)",
        n_faces, user_id, n_clusters, n_noise, elapsed,
    )

    return {
        "n_faces": n_faces,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "elapsed_s": elapsed,
    }


def cluster_all_users(db, user_ids=None, **kwargs):
    """Cluster faces for multiple users.

    Args:
        db: DB instance.
        user_ids: List of user IDs to cluster. If None, discovers all users
                  with face detections from the DB.
        **kwargs: Override HDBSCAN parameters.

    Returns:
        Dict mapping user_id to cluster stats.
    """
    results = {}
    for user_id in user_ids:
        try:
            stats = cluster_user_faces(db, user_id, **kwargs)
            results[user_id] = stats
        except Exception as e:
            log.error("Clustering failed for user %s: %s", user_id, e)
            results[user_id] = {"error": str(e)}

    return results
