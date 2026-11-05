<?php
/**
 * Fail if lib/ calls a \OC::$server->method() that the installed Nextcloud
 * no longer has.
 *
 * WHY: Nextcloud 34 removed the old Server getters. PendingService called
 * \OC::$server->getConfig(), FileListener caught the Error and only logged
 * it, so no upload was queued for classification for weeks (2,733 photos
 * missed, found 2026-11-05). php -l cannot see this; only the live class can.
 *
 * Usage: php tools/check_server_calls.php [nextcloud_root]
 */
$nc = $argv[1] ?? '/usr/share/nextcloud';
require "$nc/3rdparty/autoload.php";
require "$nc/lib/composer/autoload.php";

$bad = 0;
$root = dirname(__DIR__);
$it = new RecursiveIteratorIterator(new RecursiveDirectoryIterator("$root/lib"));
foreach ($it as $f) {
	if ($f->getExtension() !== 'php') {
		continue;
	}
	foreach (file($f->getPathname()) as $n => $line) {
		if (preg_match_all('/\\\\OC::\$server->(\w+)\s*\(/', $line, $m)) {
			foreach ($m[1] as $method) {
				if (!method_exists(\OC\Server::class, $method)) {
					printf("FAIL: %s:%d calls \\OC::\$server->%s(), which this Nextcloud does not have\n",
						substr($f->getPathname(), strlen($root) + 1), $n + 1, $method);
					$bad++;
				}
			}
		}
	}
}
echo $bad === 0 ? "ok: every \\OC::\$server call in lib/ exists in $nc\n" : '';
exit($bad === 0 ? 0 : 1);
