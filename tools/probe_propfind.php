<?php
/**
 * Drive PropFindPlugin through a real Sabre PropFind, for every child of a
 * face cluster, and report any handler that throws.
 *
 * WHY THIS EXISTS
 *   A PROPFIND on a face cluster returned HTTP 500 for weeks because
 *   PropFindPlugin called DavUtil::getDavPermissions() with one argument
 *   (Nextcloud has wanted two since 25.0). One throwing property handler
 *   fails the entire multistatus response, so the whole Faces view goes down.
 *
 *   It could not be reproduced by calling the mappers or the node methods
 *   directly — they all pass. The throw only happens inside the closure that
 *   Sabre invokes while evaluating the PropFind, which is what this drives.
 *
 * Usage:  sudo -u apache php tools/probe_propfind.php <uid> [clusterId]
 * Exit 0 if every property of every child serialises.
 */
define('OC_CONSOLE', 1);
require_once '/usr/share/nextcloud/lib/base.php';

$uid = $argv[1] ?? null;
if ($uid === null) {
    fwrite(STDERR, "usage: sudo -u apache php tools/probe_propfind.php <uid> [clusterId]\n");
    exit(2);
}
$clusterId = isset($argv[2]) ? (int)$argv[2] : 3541;

\OC_Util::setupFS($uid);
$user = \OC::$server->get(\OCP\IUserManager::class)->get($uid);
if ($user === null) { fwrite(STDERR, "no such user: $uid\n"); exit(1); }
// FacePhoto::isFavorite() calls ITagManager::load('files'), which resolves the
// SESSION user. Without this the tagger is null and every child fails on
// favorite before ever reaching the permissions handler - a CLI artifact, not
// the production fault.
\OC::$server->get(\OCP\IUserSession::class)->setUser($user);

$clusterMapper   = \OC::$server->get(\OCA\Recognize\Db\FaceClusterMapper::class);
$detectionMapper = \OC::$server->get(\OCA\Recognize\Db\FaceDetectionMapper::class);
$rootFolder      = \OC::$server->get(\OCP\Files\IRootFolder::class);
$tagManager      = \OC::$server->get(\OCP\ITagManager::class);
$previewManager  = \OC::$server->get(\OCP\IPreview::class);
$plugin          = \OC::$server->get(\OCA\Recognize\Dav\Faces\PropFindPlugin::class);

$all = $clusterMapper->findByUserId($uid);
$cluster = null;
foreach ($all as $c) {
    if ($c->getId() === $clusterId) { $cluster = $c; break; }
}
if (!$cluster) {
    // Fall back to the largest cluster rather than failing on a typo'd id.
    $best = null; $bestN = -1;
    $counts = $detectionMapper->batchCountByUser($uid);
    foreach ($all as $c) {
        $n = $counts[$c->getId()] ?? 0;
        if ($n > $bestN) { $bestN = $n; $best = $c; }
    }
    $cluster = $best;
    if (!$cluster) { fwrite(STDERR, "no clusters for $uid\n"); exit(1); }
    fwrite(STDERR, "cluster $clusterId not found; using largest instead: "
        . $cluster->getId() . " ($bestN detections)\n");
}
echo "testing cluster " . $cluster->getId() . " \"" . ($cluster->getTitle() ?? '') . "\"\n";

$root = new \OCA\Recognize\Dav\Faces\FaceRoot(
    $clusterMapper, $cluster, $user, $detectionMapper, $rootFolder, $tagManager, $previewManager
);

// The property set a real client asks for: Nextcloud's defaults plus the
// recognize-specific ones the Photos app requests.
$props = [
    '{DAV:}getcontentlength', '{DAV:}getcontenttype', '{DAV:}getetag',
    '{DAV:}getlastmodified', '{DAV:}resourcetype', '{DAV:}quota-used-bytes',
    '{http://owncloud.org/ns}fileid', '{http://owncloud.org/ns}permissions',
    '{http://owncloud.org/ns}size', '{http://owncloud.org/ns}favorite',
    '{http://nextcloud.org/ns}has-preview', '{http://nextcloud.org/ns}mount-type',
    '{http://nextcloud.org/ns}is-encrypted', '{http://nextcloud.org/ns}realpath',
    '{http://nextcloud.org/ns}face-detections',
    '{http://nextcloud.org/ns}face-preview-image',
    '{http://nextcloud.org/ns}nbItems',
];

function evaluate(string $label, $plugin, $node, array $props): bool {
    $pf = new \Sabre\DAV\PropFind('recognize/x', $props, 1);
    try {
        $plugin->propFind($pf, $node);
        $res = $pf->getResultForMultiStatus();   // forces every handler to run
        $n200 = isset($res[200]) ? count($res[200]) : 0;
        $n404 = isset($res[404]) ? count($res[404]) : 0;
        echo "  OK    $label  (200: $n200 props, 404: $n404)\n";
        return true;
    } catch (\Throwable $e) {
        echo "  THREW $label\n";
        echo "        " . get_class($e) . ': ' . $e->getMessage() . "\n";
        foreach (array_slice($e->getTrace(), 0, 5) as $f) {
            if (!empty($f['file'])) {
                echo "        " . $f['file'] . ':' . ($f['line'] ?? '?') . ' ' . ($f['function'] ?? '') . "\n";
            }
        }
        return false;
    }
}

echo "=== FaceRoot ===\n";
evaluate('FaceRoot', $plugin, $root, $props);

echo "\n=== children (every one) ===\n";
$children = $root->getChildren();
echo "  children: " . count($children) . "\n";
$ok = 0; $bad = 0; $firstBad = null;
foreach ($children as $child) {
    $pf = new \Sabre\DAV\PropFind('recognize/x', $props, 0);
    try {
        $plugin->propFind($pf, $child);
        $pf->getResultForMultiStatus();
        $ok++;
    } catch (\Throwable $e) {
        $bad++;
        if ($firstBad === null) {
            $firstBad = get_class($e) . ': ' . $e->getMessage();
            $trace = array_slice($e->getTrace(), 0, 5);
        }
    }
}
echo "  serialised OK : $ok\n";
echo "  FAILED        : $bad\n";
if ($firstBad !== null) {
    echo "\n  first failure: $firstBad\n";
    foreach ($trace as $f) {
        if (!empty($f['file'])) echo "      " . $f['file'] . ':' . ($f['line'] ?? '?') . ' ' . ($f['function'] ?? '') . "\n";
    }
}
echo "\n" . ($bad === 0 ? "PASS - the whole PROPFIND serialises\n" : "FAIL\n");
exit($bad === 0 ? 0 : 1);
