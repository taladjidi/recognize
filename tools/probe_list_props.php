<?php
/**
 * Time each property of the Faces overview PROPFIND, server-side, and count
 * the DB queries it runs.
 *
 * WHY: bench_faces.py showed the overview list taking ~4.6 s, ~4.2 s of it
 * only when nc:nbItems is requested, while the batched count query behind it
 * takes ~70 ms. This drives the real FacesHome::getChildren and the real
 * PropFindPlugin for every listed cluster, one property at a time, so the
 * query count shows whether something falls back to per-cluster lookups.
 *
 * Usage: sudo -u apache php tools/probe_list_props.php <uid>     (read-only)
 */
define('OC_CONSOLE', 1);
require_once '/usr/share/nextcloud/lib/base.php';

$uid = $argv[1] ?? null;
if ($uid === null) {
	fwrite(STDERR, "usage: sudo -u apache php tools/probe_list_props.php <uid>\n");
	exit(2);
}
\OC_Util::setupFS($uid);
$user = \OC::$server->get(\OCP\IUserManager::class)->get($uid);
\OC::$server->get(\OCP\IUserSession::class)->setUser($user);

$s = \OC::$server;
$db = $s->get(\OCP\IDBConnection::class)->getInner();
$plugin = $s->get(\OCA\Recognize\Dav\Faces\PropFindPlugin::class);
$home = new \OCA\Recognize\Dav\Faces\FacesHome(
	$s->get(\OCA\Recognize\Db\FaceClusterMapper::class), $user,
	$s->get(\OCA\Recognize\Db\FaceDetectionMapper::class), $s->get(\OCP\Files\IRootFolder::class),
	$s->get(\OCP\ITagManager::class), $s->get(\OCP\IPreview::class));

function measure(string $label, callable $fn, $db): void {
	$q0 = $db->getStats()['executed'];
	$t0 = hrtime(true);
	$n = $fn();
	printf("  %-28s %8.1f ms  %6d queries  %s\n", $label, (hrtime(true) - $t0) / 1e6, $db->getStats()['executed'] - $q0, $n);
}

$children = [];
measure('getChildren', function () use ($home, &$children) { $children = $home->getChildren(); return count($children) . ' clusters'; }, $db);

foreach ([
	'nbItems' => '{http://nextcloud.org/ns}nbItems',
	'face-preview-image' => '{http://nextcloud.org/ns}face-preview-image',
	'displayname' => '{DAV:}displayname',
	'getetag' => '{DAV:}getetag',
	'fileid' => '{http://owncloud.org/ns}fileid',
	'permissions' => '{http://owncloud.org/ns}permissions',
	'favorite' => '{http://owncloud.org/ns}favorite',
] as $label => $prop) {
	measure($label, function () use ($children, $plugin, $prop) {
		$found = 0;
		foreach ($children as $child) {
			$pf = new \Sabre\DAV\PropFind('recognize/x', [$prop], 0);
			$plugin->propFind($pf, $child);
			$found += count($pf->getResultForMultiStatus()[200] ?? []);
		}
		return "$found with a value";
	}, $db);
}

// The same, via the node methods Sabre's core plugin calls for every child.
measure('getLastModified (node)', function () use ($children) { foreach ($children as $c) { $c->getLastModified(); } return ''; }, $db);
measure('getName (node)', function () use ($children) { foreach ($children as $c) { $c->getName(); } return ''; }, $db);
