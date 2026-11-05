<?php
/**
 * Parse appinfo/info.xml with Nextcloud's own InfoParser and fail if any
 * repair-steps list is not an array. An empty <post-migration></post-migration>
 * parses as a whitespace string and makes `occ upgrade` throw a TypeError in
 * AppManager::executeRepairSteps, leaving the instance in maintenance mode
 * (happened 2026-11-05).
 *
 * Usage: php tools/check_info_xml.php [path/to/info.xml]
 */
require '/usr/share/nextcloud/lib/private/App/InfoParser.php';
$path = $argv[1] ?? __DIR__ . '/../appinfo/info.xml';
$info = (new OC\App\InfoParser())->parse($path);
if (!is_array($info)) {
	fwrite(STDERR, "FAIL: $path does not parse\n");
	exit(1);
}
$bad = array_filter(['install', 'pre-migration', 'post-migration', 'live-migration'],
	fn ($k) => !is_array($info['repair-steps'][$k]));
if ($bad) {
	fwrite(STDERR, "FAIL: $path repair-steps not a list: " . implode(', ', $bad) . " (remove empty elements)\n");
	exit(1);
}
echo "ok: $path\n";
