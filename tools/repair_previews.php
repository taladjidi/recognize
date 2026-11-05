<?php
/**
 * Drop stale preview records whose file cannot be read, so Nextcloud regenerates them.
 *
 * WHY: for some large exports the preview index (oc_previews) listed a "max" preview
 * (e.g. 1365-2048-max.jpg) whose file was missing; Generator then tries to read it
 * instead of generating, throws NotFoundException, and Photos answers 404 forever
 * (tools/probe_preview.php shows it). A preview generation interrupted or failed earlier
 * leaves such a row behind.
 *
 * For each file id: list its indexed previews, try to open each; if any is unreadable,
 * delete ALL of that file's previews (larger sizes are derived from the max one); they
 * are regenerated on the next request. Dry run unless --apply.
 *
 * Usage: sudo -u apache php tools/repair_previews.php [--apply] <fileid> [<fileid> ...]
 */
define('OC_CONSOLE', 1);
require_once '/usr/share/nextcloud/lib/base.php';

$args = array_slice($argv, 1);
$apply = in_array('--apply', $args, true);
$ids = array_map('intval', array_filter($args, fn ($a) => $a !== '--apply'));
if ($ids === [] || in_array(0, $ids, true)) {
	fwrite(STDERR, "usage: sudo -u apache php tools/repair_previews.php [--apply] <fileid> [<fileid> ...]\n");
	exit(2);
}
$service = \OC::$server->get(\OC\Preview\PreviewService::class);
$storage = \OC::$server->get(\OC\Preview\Storage\StorageFactory::class);
$mapper = \OC::$server->get(\OC\Preview\Db\PreviewMapper::class);

$bad = 0;
foreach ($ids as $fileId) {
	$previews = iterator_to_array($service->getAvailablePreviewsForFile($fileId), false);
	$unreadable = [];
	foreach ($previews as $p) {
		try {
			$h = $storage->readPreview($p);
			if (is_resource($h)) {
				fclose($h);
			}
		} catch (\Throwable $e) {
			$unreadable[] = $p->getName();
		}
	}
	printf("file %d: %d indexed previews, %d unreadable%s\n", $fileId, count($previews), count($unreadable),
		$unreadable ? ' (' . implode(', ', $unreadable) . ')' : '');
	if ($unreadable === []) {
		continue;
	}
	$bad++;
	if (!$apply) {
		echo "  would delete all " . count($previews) . " previews of file $fileId\n";
		continue;
	}
	foreach ($previews as $p) {
		try {
			$service->deletePreview($p);  // stored file + row
		} catch (\Throwable $e) {
			$mapper->delete($p);          // file already gone: drop the row only
		}
	}
	echo "  deleted; the next request regenerates them\n";
}
echo ($apply ? 'applied' : 'dry run') . ": $bad of " . count($ids) . " files had unreadable previews\n";
