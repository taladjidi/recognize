<?php
/**
 * Ask Nextcloud's preview manager for one file's preview and print why it fails.
 *
 * WHY: the Photos preview endpoint answers 404 for some large JPEG exports and
 * nextcloud.log records only "Requesting preview ...": the real exception is swallowed.
 * Read-only apart from the preview it may generate.
 *
 * Usage: sudo -u apache php tools/probe_preview.php <uid> <fileid> [size]
 */
define('OC_CONSOLE', 1);
require_once '/usr/share/nextcloud/lib/base.php';

[$uid, $fileId, $size] = [$argv[1] ?? null, (int)($argv[2] ?? 0), (int)($argv[3] ?? 1024)];
if ($uid === null || $fileId <= 0) {
	fwrite(STDERR, "usage: sudo -u apache php tools/probe_preview.php <uid> <fileid> [size]\n");
	exit(2);
}
\OC_Util::setupFS($uid);
$s = \OC::$server;
$node = $s->get(\OCP\Files\IRootFolder::class)->getUserFolder($uid)->getFirstNodeById($fileId);
if ($node === null) {
	echo "no node $fileId for $uid\n";
	exit(1);
}
$preview = $s->get(\OCP\IPreview::class);
$config = $s->get(\OCP\IConfig::class);
echo "file: {$node->getPath()}  mime {$node->getMimetype()}  " . round($node->getSize() / 1048576, 1) . " MB\n";
foreach (['preview_max_filesize_image', 'preview_max_memory', 'preview_max_x', 'preview_max_y', 'enable_previews'] as $k) {
	echo "  $k = " . json_encode($config->getSystemValue($k, '(default)')) . "\n";
}
echo "  memory_limit = " . ini_get('memory_limit') . "  (CLI; php-fpm may differ)\n";
echo "  isMimeSupported: " . var_export($preview->isMimeSupported($node->getMimetype()), true)
	. "  isAvailable: " . var_export($preview->isAvailable($node), true) . "\n";
if (function_exists('getimagesize') && $node instanceof \OCP\Files\File) {
	$local = $node->getStorage()->getLocalFile($node->getInternalPath());
	$info = $local ? @getimagesize($local) : false;
	echo '  getimagesize: ' . ($info ? "{$info[0]}x{$info[1]} channels=" . ($info['channels'] ?? '?') . ' bits=' . ($info['bits'] ?? '?') . " {$info['mime']}" : 'failed') . "\n";
}
$t0 = hrtime(true);
try {
	$p = $preview->getPreview($node, $size, $size, false);
	printf("OK: %s %dx? in %.0f ms\n", $p->getMimeType(), $size, (hrtime(true) - $t0) / 1e6);
} catch (\Throwable $e) {
	printf("FAILED after %.0f ms: %s: %s\n", (hrtime(true) - $t0) / 1e6, get_class($e), $e->getMessage());
	foreach (array_slice($e->getTrace(), 0, 6) as $f) {
		echo '    ' . ($f['file'] ?? '?') . ':' . ($f['line'] ?? '?') . ' ' . ($f['class'] ?? '') . ($f['type'] ?? '') . ($f['function'] ?? '') . "\n";
	}
}
echo '  peak memory ' . round(memory_get_peak_usage(true) / 1048576) . " MB\n";
