<?php

declare(strict_types=1);

// Prints a Faces API token minted by the PHP code, for python/tests/test_api_server.py.
// Usage: php tools/check_token.php <secret> <uid> <exp>

require __DIR__ . '/../lib/Service/FacesToken.php';

if ($argc !== 4) {
	fwrite(STDERR, "usage: php check_token.php <secret> <uid> <exp>\n");
	exit(2);
}
echo \OCA\Recognize\Service\FacesToken::mint($argv[1], $argv[2], (int)$argv[3]);
