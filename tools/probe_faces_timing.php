<?php
/**
 * Time the server work behind the Faces overview and one cluster. Read-only.
 *
 * Pairs with tools/bench_faces.py: the bench shows what the client waits for,
 * this shows which part of it is the database and which is PHP.
 *
 * Usage:  sudo -u apache php tools/probe_faces_timing.php <uid> [clusterId]
 */
define('OC_CONSOLE', 1);
require_once '/usr/share/nextcloud/lib/base.php';

$uid = $argv[1] ?? null;
if ($uid === null) {
    fwrite(STDERR, "usage: sudo -u apache php tools/probe_faces_timing.php <uid> [clusterId]\n");
    exit(2);
}
\OC_Util::setupFS($uid);
$user = \OC::$server->get(\OCP\IUserManager::class)->get($uid);
if ($user === null) { fwrite(STDERR, "no such user: $uid\n"); exit(1); }
// FacePhoto::isFavorite() resolves the session user (see probe_propfind.php).
\OC::$server->get(\OCP\IUserSession::class)->setUser($user);

$db              = \OC::$server->get(\OCP\IDBConnection::class);
$clusterMapper   = \OC::$server->get(\OCA\Recognize\Db\FaceClusterMapper::class);
$detectionMapper = \OC::$server->get(\OCA\Recognize\Db\FaceDetectionMapper::class);
$rootFolder      = \OC::$server->get(\OCP\Files\IRootFolder::class);
$tagManager      = \OC::$server->get(\OCP\ITagManager::class);
$previewManager  = \OC::$server->get(\OCP\IPreview::class);

function timed(string $label, callable $fn) {
    $t = hrtime(true);
    $r = $fn();
    printf("  %-44s %9.1f ms  %s rows\n", $label, (hrtime(true) - $t) / 1e6, is_countable($r) ? count($r) : '?');
    return $r;
}

echo "=== overview ===\n";
$clusters = timed('FaceClusterMapper::findByUserId', fn () => $clusterMapper->findByUserId($uid));
$counts   = timed('FaceDetectionMapper::batchCountByUser', fn () => $detectionMapper->batchCountByUser($uid));
timed('FaceDetectionMapper::batchPreviewsByUser', fn () => $detectionMapper->batchPreviewsByUser($uid));

// Largest cluster by detection count (or the id given).
$clusterId = isset($argv[2]) ? (int)$argv[2] : null;
if ($clusterId === null) {
    arsort($counts);
    $clusterId = (int)array_key_first($counts);
}
$cluster = null;
foreach ($clusters as $c) { if ($c->getId() === $clusterId) { $cluster = $c; break; } }
if ($cluster === null) { fwrite(STDERR, "cluster $clusterId not found\n"); exit(1); }

echo "\n=== cluster $clusterId (" . ($counts[$clusterId] ?? 0) . " detections) ===\n";
timed('findByClusterIdWithExistingFiles', fn () => $detectionMapper->findByClusterIdWithExistingFiles($clusterId, $uid));
// Fresh FaceRoot so nothing is cached; getChildren() is what a PROPFIND Depth:1 calls.
$root = new \OCA\Recognize\Dav\Faces\FaceRoot(
    $clusterMapper, $cluster, $user, $detectionMapper, $rootFolder, $tagManager, $previewManager
);
timed('FaceRoot::getChildren', fn () => $root->getChildren());

// EXPLAIN the SQL the mappers run. Rebuilt here from the same query-builder
// calls as lib/Db (keep in sync); getSQL() gives the prefixed table names.
echo "\n=== EXPLAIN ===\n";
$queries = [];

$qb = $db->getQueryBuilder();
$cols = array_map(fn ($c) => 'fc.' . $c, \OCA\Recognize\Db\FaceCluster::$columns);
$qb->selectDistinct($cols)->from('recognize_face_clusters', 'fc')
    ->innerJoin('fc', 'recognize_face_detections', 'fd', $qb->expr()->andX(
        $qb->expr()->eq('fc.id', 'fd.cluster_id'),
        $qb->expr()->eq('fd.user_id', $qb->createPositionalParameter($uid))))
    ->where($qb->expr()->eq('fc.user_id', $qb->createPositionalParameter($uid)));
$queries['findByUserId'] = $qb;

$base = function (array $select) use ($db, $uid) {
    $qb = $db->getQueryBuilder();
    $qb->select(...$select)->from('recognize_face_detections', 'd')
        ->innerJoin('d', 'filecache', 'fc', $qb->expr()->eq('d.file_id', 'fc.fileid'))
        ->where($qb->expr()->eq('d.user_id', $qb->createPositionalParameter($uid)))
        ->andWhere($qb->expr()->like('fc.path', $qb->createPositionalParameter('files/%')));
    return $qb;
};
$qb = $base(['d.cluster_id', $db->getQueryBuilder()->func()->count('d.id', 'cnt')]);
$qb->andWhere($qb->expr()->isNotNull('d.cluster_id'))
    ->andWhere($qb->expr()->gt('d.cluster_id', $qb->createPositionalParameter(0)))
    ->groupBy('d.cluster_id');
$queries['batchCountByUser'] = $qb;

$qb = $base(['d.*']);
$qb->andWhere($qb->expr()->isNotNull('d.cluster_id'))
    ->andWhere($qb->expr()->gt('d.cluster_id', $qb->createPositionalParameter(0)))
    ->addOrderBy('d.cluster_id')->addOrderBy('d.height', 'DESC')->addOrderBy('d.width', 'DESC');
$queries['batchPreviewsByUser'] = $qb;

$qb = $base(['d.*']);
$qb->andWhere($qb->expr()->eq('d.cluster_id', $qb->createPositionalParameter($clusterId)));
$queries['findByClusterIdWithExistingFiles'] = $qb;

foreach ($queries as $name => $qb) {
    $sql = $qb->getSQL();
    echo "\n-- $name\n$sql\n";
    try {
        // Positional params are all strings/ints; bind them so EXPLAIN is valid.
        $stmt = $db->prepare('EXPLAIN ' . $sql);
        $i = 1;
        foreach ($qb->getParameters() as $v) { $stmt->bindValue($i++, $v); }
        $res = $stmt->execute();
        while ($row = $res->fetch()) {
            echo '   ' . json_encode($row) . "\n";
        }
        $res->closeCursor();
    } catch (\Throwable $e) {
        echo '   EXPLAIN failed: ' . $e->getMessage() . "\n";
    }
}
