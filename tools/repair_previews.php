<?php
/**
 * Drop stale preview records whose file cannot be read, so Nextcloud regenerates them.
 *
 * WHY: oc_previews can list a preview whose file is gone. Generator then serves the row
 * instead of generating, the read fails while the response is being sent, and the client
 * gets an HTML error page with HTTP 200 / image/jpeg (Photos shows an empty tile), or a
 * 404 when it was the "max" preview. Cause found 2026-11-06: before 34.0.5, the loser of
 * two concurrent generations of the same preview deleted the shared file
 * (LocalPreviewStorage::deleteUnreferencedPreview, now a no-op), leaving the winner's
 * row without a file.
 *
 * Per file: a row whose file is unreadable is deleted (its file is already gone). If the
 * "max" preview is among them, all of that file's previews go (the other sizes are
 * derived from it); they are regenerated on the next request. Dry run unless --apply.
 *
 * Usage: sudo -u apache php tools/repair_previews.php [--apply] <fileid> [<fileid> ...]
 *        sudo -u apache php tools/repair_previews.php [--apply] --all     (every row)
 */
define('OC_CONSOLE', 1);
require_once '/usr/share/nextcloud/lib/base.php';

$args = array_slice($argv, 1);
$apply = in_array('--apply', $args, true);
$all = in_array('--all', $args, true);
$ids = array_map('intval', array_values(array_filter($args, fn ($a) => !str_starts_with($a, '--'))));
if ((!$all && $ids === []) || in_array(0, $ids, true) || ($all && $ids !== [])) {
	fwrite(STDERR, "usage: sudo -u apache php tools/repair_previews.php [--apply] (<fileid> ... | --all)\n");
	exit(2);
}
$service = \OC::$server->get(\OC\Preview\PreviewService::class);
$storage = \OC::$server->get(\OC\Preview\Storage\StorageFactory::class);
$mapper = \OC::$server->get(\OC\Preview\Db\PreviewMapper::class);

$readable = function (\OC\Preview\Db\Preview $p) use ($storage): bool {
	try {
		$h = $storage->readPreview($p);
		if (is_resource($h)) {
			fclose($h);
		}
		return true;
	} catch (\Throwable $e) {
		return false;
	}
};

/** Repair one file's rows; returns [rows deleted, whether the whole file was reset]. */
$repair = function (int $fileId, array $previews, array $bad) use ($apply, $service, $mapper, $readable): array {
	$maxGone = (bool)array_filter($bad, fn ($p) => $p->isMax());
	$targets = $maxGone ? $previews : $bad;
	if (!$apply) {
		return [count($targets), $maxGone];
	}
	$n = 0;
	foreach ($targets as $p) {
		if (!$maxGone && $readable($p)) {
			continue;  // regenerated since the scan: keep it
		}
		try {
			$service->deletePreview($p);  // stored file (if any) + row
		} catch (\Throwable $e) {
			$mapper->delete($p);          // file already gone: drop the row only
		}
		$n++;
	}
	return [$n, $maxGone];
};

$stats = ['rows' => 0, 'files' => 0, 'bad_rows' => 0, 'bad_files' => 0, 'max_gone' => 0, 'deleted' => 0];
$byMonth = [];
$t0 = microtime(true);

$handle = function (int $fileId, array $previews, bool $verbose) use (&$stats, &$byMonth, $readable, $repair) {
	$stats['files']++;
	$stats['rows'] += count($previews);
	$bad = array_values(array_filter($previews, fn ($p) => !$readable($p)));
	if ($verbose) {
		printf("file %d: %d indexed previews, %d unreadable%s\n", $fileId, count($previews), count($bad),
			$bad ? ' (' . implode(', ', array_map(fn ($p) => $p->getName(), $bad)) . ')' : '');
	}
	if ($bad === []) {
		return;
	}
	$stats['bad_rows'] += count($bad);
	$stats['bad_files']++;
	foreach ($bad as $p) {
		$m = date('Y-m', $p->getMtime());
		$byMonth[$m] = ($byMonth[$m] ?? 0) + 1;
	}
	[$n, $maxGone] = $repair($fileId, $previews, $bad);
	$stats['deleted'] += $n;
	$stats['max_gone'] += $maxGone ? 1 : 0;
};

if ($all) {
	// Stream rows in id order (404k entities do not fit in memory); keep only the bad ones.
	$badByFile = [];
	$lastId = '0';
	$seen = [];
	do {
		$i = 0;
		foreach ($mapper->getPreviews($lastId, \OC\Preview\Db\PreviewMapper::MAX_CHUNK_SIZE) as $p) {
			$lastId = (string)$p->getId();
			$i++;
			$stats['rows']++;
			$seen[$p->getFileId()] = true;
			if (!$readable($p)) {
				$badByFile[$p->getFileId()][] = $p;
			}
		}
		fprintf(STDERR, "\r  checked %d rows, %d unreadable (%.0f s)", $stats['rows'], array_sum(array_map('count', $badByFile)), microtime(true) - $t0);
	} while ($i === \OC\Preview\Db\PreviewMapper::MAX_CHUNK_SIZE);
	fprintf(STDERR, "\n");
	$stats['files'] = count($seen);
	unset($seen);
	foreach ($badByFile as $fileId => $bad) {
		$stats['bad_rows'] += count($bad);
		$stats['bad_files']++;
		foreach ($bad as $p) {
			$m = date('Y-m', $p->getMtime());
			$byMonth[$m] = ($byMonth[$m] ?? 0) + 1;
		}
		$previews = iterator_to_array($service->getAvailablePreviewsForFile($fileId), false);
		[$n, $maxGone] = $repair($fileId, $previews, $bad);
		$stats['deleted'] += $n;
		$stats['max_gone'] += $maxGone ? 1 : 0;
	}
} else {
	foreach ($ids as $fileId) {
		$handle($fileId, iterator_to_array($service->getAvailablePreviewsForFile($fileId), false), true);
	}
}

ksort($byMonth);
echo "unreadable rows by month the row was written:\n";
foreach ($byMonth as $m => $n) {
	echo "  $m  $n\n";
}
printf("%s: %d rows in %d files checked in %.0f s; %d unreadable rows in %d files (%d with the max preview gone); %s %d rows\n",
	$apply ? 'applied' : 'dry run', $stats['rows'], $stats['files'], microtime(true) - $t0,
	$stats['bad_rows'], $stats['bad_files'], $stats['max_gone'], $apply ? 'deleted' : 'would delete', $stats['deleted']);
