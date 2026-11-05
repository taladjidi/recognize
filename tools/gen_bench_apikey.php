<?php
/**
 * Print a fresh X-Recognize-Api-Key (valid 24 h) for tools/bench_faces.py.
 * The browser gets one from the Photos page state and sends it on every DAV
 * request; with Basic auth the recognize DAV plugin runs before the user is
 * known, so the bench needs one too.
 *
 *   (umask 077; sudo -u apache php tools/gen_bench_apikey.php > ~/.config/nc-bench/recognize-api-key)
 */
define('OC_CONSOLE', 1);
require_once '/usr/share/nextcloud/lib/base.php';
echo \OC::$server->get(\OCA\Recognize\Public\ApiKeyManager::class)->generateApiKey(), "\n";
