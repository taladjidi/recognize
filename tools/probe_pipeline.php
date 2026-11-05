<?php
/**
 * Is every photo and video classified, and does a new one get picked up?
 *
 *   sudo -u apache php tools/probe_pipeline.php coverage [uid ...]
 *       Per user: media files under files/, how many carry the recognize
 *       "processed" tag, how many sit in recognize_pending, and how many
 *       are neither (missed). Prints up to 10 missed paths. Read-only.
 *
 *   sudo -u apache php tools/probe_pipeline.php e2e <uid> [timeout_s]
 *       Creates <uid>/files/RecognizeProbe/probe-<ts>.jpg through the Files
 *       API (the same NodeCreatedEvent an upload fires), then reports when
 *       it is queued and when the daemon tags it, the labels and faces
 *       found, and deletes it again. Exit 0 only if it got tagged in time.
 *
 * WHY THIS EXISTS: the daemon's journal only says "Processed N images"; it
 * cannot show a file that never reached the queue. Uploads are queued by
 * FileListener, but files that appear any other way (occ files:scan, external
 * storage, a sync tool writing into the data dir) fire no event and are only
 * found by a StorageCrawlJob.
 *
 * shortcut: raw SQL with CAST(... AS CHAR) is MySQL/MariaDB-only (this host);
 * port to the query builder if it is ever run against PostgreSQL.
 */
define('OC_CONSOLE', 1);
require_once '/usr/share/nextcloud/lib/base.php';

use OCA\Recognize\Constants;
use OCA\Recognize\Service\TagManager;

$mode = $argv[1] ?? '';
if (!in_array($mode, ['coverage', 'e2e'], true)) {
	fwrite(STDERR, "usage: sudo -u apache php tools/probe_pipeline.php coverage [uid ...] | e2e <uid> [timeout_s]\n");
	exit(2);
}

$server = \OC::$server;
$db = $server->get(\OCP\IDBConnection::class);
$rootFolder = $server->get(\OCP\Files\IRootFolder::class);
$mimeLoader = $server->get(\OCP\Files\IMimeTypeLoader::class);
$processedTagId = (int)$server->get(TagManager::class)->getProcessedTag()->getId();

function storageIdOf(string $uid): int {
	\OC_Util::setupFS($uid);
	return (int)\OC::$server->get(\OCP\Files\IRootFolder::class)->getUserFolder($uid)
		->getStorage()->getCache()->getNumericStorageId();
}

function isTagged(\OCP\IDBConnection $db, int $fileId, int $tagId): bool {
	return (bool)$db->executeQuery(
		'SELECT 1 FROM `*PREFIX*systemtag_object_mapping` WHERE `objecttype` = ? AND `objectid` = ? AND `systemtagid` = ?',
		['files', (string)$fileId, $tagId])->fetchOne();
}

function isPending(\OCP\IDBConnection $db, int $fileId): bool {
	return (bool)$db->executeQuery('SELECT 1 FROM `*PREFIX*recognize_pending` WHERE `file_id` = ?', [$fileId])->fetchOne();
}

if ($mode === 'coverage') {
	$uids = array_slice($argv, 2);
	if ($uids === []) {
		$server->get(\OCP\IUserManager::class)->callForSeenUsers(function ($u) use (&$uids) { $uids[] = $u->getUID(); });
	}
	$mimeIds = array_values(array_filter(array_map(
		fn ($m) => $mimeLoader->exists($m) ? $mimeLoader->getId($m) : null,
		array_merge(Constants::IMAGE_FORMATS, Constants::VIDEO_FORMATS))));
	$in = implode(',', array_map('intval', $mimeIds));
	$anyMissed = false;
	foreach ($uids as $uid) {
		$storage = storageIdOf($uid);
		$row = $db->executeQuery(
			"SELECT COUNT(*) AS total,
			        SUM(t.objectid IS NOT NULL) AS tagged,
			        SUM(t.objectid IS NULL AND p.file_id IS NOT NULL) AS pending,
			        SUM(t.objectid IS NULL AND p.file_id IS NULL) AS missed,
			        MAX(CASE WHEN t.objectid IS NULL AND p.file_id IS NULL THEN f.mtime END) AS newest_missed
			 FROM `*PREFIX*filecache` f
			 LEFT JOIN `*PREFIX*systemtag_object_mapping` t
			   ON t.objecttype = 'files' AND t.objectid = CAST(f.fileid AS CHAR) AND t.systemtagid = ?
			 LEFT JOIN `*PREFIX*recognize_pending` p ON p.file_id = f.fileid
			 WHERE f.storage = ? AND f.path LIKE 'files/%' AND f.mimetype IN ($in)",
			[$processedTagId, $storage])->fetch();
		printf("%-12s %6d media  %6d tagged  %5d pending  %5d missed%s\n", $uid,
			$row['total'], $row['tagged'], $row['pending'], $row['missed'],
			$row['missed'] > 0 ? '  (newest missed: ' . date('Y-m-d H:i', (int)$row['newest_missed']) . ')' : '');
		if ($row['missed'] > 0) {
			$anyMissed = true;
			$res = $db->executeQuery(
				"SELECT f.path, f.mtime FROM `*PREFIX*filecache` f
				 LEFT JOIN `*PREFIX*systemtag_object_mapping` t
				   ON t.objecttype = 'files' AND t.objectid = CAST(f.fileid AS CHAR) AND t.systemtagid = ?
				 LEFT JOIN `*PREFIX*recognize_pending` p ON p.file_id = f.fileid
				 WHERE f.storage = ? AND f.path LIKE 'files/%' AND f.mimetype IN ($in)
				   AND t.objectid IS NULL AND p.file_id IS NULL
				 ORDER BY f.mtime DESC LIMIT 10", [$processedTagId, $storage]);
			while ($r = $res->fetch()) {
				printf("    %s  %s\n", date('Y-m-d', (int)$r['mtime']), $r['path']);
			}
		}
	}
	exit($anyMissed ? 1 : 0);
}

/** Walk the listener's path step by step to find where a new file is lost. */
function diagnose(\OCP\Files\File $file, \OCP\IDBConnection $db): void {
	$server = \OC::$server;
	$fileId = $file->getId();
	echo "  --- diagnose\n";

	$event = \OCP\Files\Events\Node\NodeCreatedEvent::class;
	$classes = [];
	foreach ($server->get(\OCP\EventDispatcher\IEventDispatcher::class)->getSymfonyDispatcher()->getListeners($event) as $l) {
		if ($l instanceof \OC\EventDispatcher\ServiceEventListener) {
			$p = new \ReflectionProperty($l, 'class');
			$classes[] = $p->getValue($l);
		} else {
			$classes[] = is_object($l) ? get_class($l) : 'callable';
		}
	}
	$registered = in_array(\OCA\Recognize\Hooks\FileListener::class, $classes, true);
	echo '  1. FileListener registered for NodeCreatedEvent: ' . ($registered ? 'yes' : 'NO') . ' (' . count($classes) . " listeners)\n";

	$listener = $server->get(\OCA\Recognize\Hooks\FileListener::class);
	$listener->handle(new \OCP\Files\Events\Node\NodeCreatedEvent($file));
	$direct = isPending($db, $fileId);
	echo '  2. calling FileListener::handle directly queues it: ' . ($direct ? 'yes -> the event is not reaching it' : 'NO -> the listener drops it') . "\n";
	if ($direct) {
		$db->executeStatement('DELETE FROM `*PREFIX*recognize_pending` WHERE `file_id` = ?', [$fileId]);
		return;
	}

	foreach (['isSupportedMimetype' => [$file->getMimetype()], 'isFileIgnored' => [$file]] as $method => $args) {
		$m = new \ReflectionMethod($listener, $method);
		try {
			echo "  3. $method(" . (is_string($args[0]) ? $args[0] : $file->getPath()) . '): ' . var_export($m->invokeArgs($listener, $args), true) . "\n";
		} catch (\Throwable $e) {
			echo "  3. $method THREW " . get_class($e) . ': ' . $e->getMessage() . "\n";
		}
	}
	echo '     storage id: ' . var_export($file->getMountPoint()->getNumericStorageId(), true) . "\n";
	try {
		$server->get(\OCA\Recognize\Service\PendingService::class)
			->upsert($fileId, $file->getMountPoint()->getNumericStorageId(), \OCA\Recognize\Service\PendingService::ACTION_CREATED);
		echo '  4. PendingService::upsert directly: ' . (isPending($db, $fileId) ? 'queued' : 'returned without queueing') . "\n";
		$db->executeStatement('DELETE FROM `*PREFIX*recognize_pending` WHERE `file_id` = ?', [$fileId]);
	} catch (\Throwable $e) {
		echo '  4. PendingService::upsert THREW ' . get_class($e) . ': ' . $e->getMessage() . "\n";
	}
}

// e2e
$uid = $argv[2] ?? null;
$timeout = (int)($argv[3] ?? 300);
if ($uid === null) {
	fwrite(STDERR, "e2e needs a uid\n");
	exit(2);
}
storageIdOf($uid);
$userFolder = $rootFolder->getUserFolder($uid);
$dir = $userFolder->nodeExists('RecognizeProbe') ? $userFolder->get('RecognizeProbe') : $userFolder->newFolder('RecognizeProbe');
$image = __DIR__ . '/../tests/res/casey-lee-awj7sRviVXo-unsplash.jpg';
$file = $dir->newFile('probe-' . date('Ymd-His') . '.jpg', file_get_contents($image));
$fileId = $file->getId();
echo "created {$file->getPath()} (fileid $fileId)\n";

$t0 = microtime(true);
$queuedAt = $taggedAt = null;
while (microtime(true) - $t0 < $timeout) {
	$elapsed = microtime(true) - $t0;
	if ($queuedAt === null && isPending($db, $fileId)) {
		$queuedAt = $elapsed;
		printf("  %5.1fs  queued in recognize_pending\n", $elapsed);
	}
	if (isTagged($db, $fileId, $processedTagId)) {
		$taggedAt = $elapsed;
		printf("  %5.1fs  tagged as processed by the daemon\n", $elapsed);
		break;
	}
	if ($queuedAt === null && $elapsed > 10) {
		echo "  NOT QUEUED after 10s: FileListener did not fire for this file\n";
		diagnose($file, $db);
		break;
	}
	sleep(2);
}

$tags = $server->get(\OCP\SystemTag\ISystemTagObjectMapper::class)->getTagIdsForObjects([(string)$fileId], 'files')[(string)$fileId] ?? [];
$names = $tags === [] ? [] : array_map(fn ($t) => $t->getName(), $server->get(\OCP\SystemTag\ISystemTagManager::class)->getTagsByIds($tags));
$faces = (int)$db->executeQuery('SELECT COUNT(*) FROM `*PREFIX*recognize_face_detections` WHERE `file_id` = ?', [$fileId])->fetchOne();
echo '  labels: ' . (implode(', ', array_diff($names, [TagManager::RECOGNIZED_TAG])) ?: '(none)') . "; faces detected: $faces\n";

$file->delete();
if ($dir->getDirectoryListing() === []) {
	$dir->delete();
}
echo "  cleaned up (the probe file is in $uid's trash bin)\n";

if ($taggedAt === null) {
	echo $queuedAt === null ? "FAIL: never queued\n" : "FAIL: queued but not processed within {$timeout}s\n";
	exit(1);
}
printf("ok: picked up and processed in %.0fs\n", $taggedAt);
