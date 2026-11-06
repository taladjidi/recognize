<?php
/**
 * Check that lib/ still fits a Nextcloud source tree (host install or a
 * checkout of stable35 / master), so max-version in info.xml is not a guess.
 *
 * Checks, per tree: every imported OCP/OC/OCA\DAV/Sabre class exists; every
 * Foo::CONST / Foo::method() on an imported class exists; every
 * $this->prop->method() on a constructor-typed OCP property exists; every lib
 * class loads (a changed interface or signature is a compile-time fatal, so
 * each loads in its own process); controllers do not rely on annotations.
 *
 * Usage: php tools/check_api_compat.php [nextcloud_root]   (default: host)
 * Sabre comes from the host's 3rdparty (a checkout has no 3rdparty).
 */
$nc = rtrim($argv[1] ?? '/usr/share/nextcloud', '/');
$root = dirname(__DIR__);

function registerNc(string $nc): void {
	$map = ['OCP\\' => "$nc/lib/public/", 'OC\\' => "$nc/lib/private/", 'OCA\\DAV\\' => "$nc/apps/dav/lib/"];
	spl_autoload_register(static function (string $c) use ($map) {
		foreach ($map as $p => $dir) {
			if (str_starts_with($c, $p) && is_file($f = $dir . str_replace('\\', '/', substr($c, strlen($p))) . '.php')) {
				require_once $f;
				return;
			}
		}
	});
}

require '/usr/share/nextcloud/3rdparty/autoload.php';
registerNc($nc);

if (($argv[2] ?? '') === '--load') { // child: compile one file against the tree
	spl_autoload_register(static function (string $c) use ($argv) {
		if (str_starts_with($c, 'OCA\\Recognize\\')) {
			$f = dirname(__DIR__) . '/lib/' . str_replace('\\', '/', substr($c, 14)) . '.php';
			is_file($f) && require_once $f;
		}
	});
	require $argv[3];
	exit(0);
}

$ext = static fn(string $c) => (bool)preg_match('/^(OCP|OC|OCA\\\\DAV|Sabre)\\\\/', $c);
$exists = static fn(string $c) => class_exists($c) || interface_exists($c) || trait_exists($c) || enum_exists($c);
$bad = 0;
$fail = static function (string $m) use (&$bad) {
	echo "FAIL: $m\n";
	$bad++;
};

$it = new RecursiveIteratorIterator(new RecursiveDirectoryIterator("$root/lib"));
foreach ($it as $f) {
	if ($f->getExtension() !== 'php') {
		continue;
	}
	$rel = substr($f->getPathname(), strlen($root) + 1);
	$src = file_get_contents($f->getPathname());

	$alias = [];
	preg_match_all('/^use\s+([\w\\\\]+)(?:\s+as\s+(\w+))?;/m', $src, $m, PREG_SET_ORDER);
	foreach ($m as $u) {
		$alias[$u[2] ?? substr(strrchr('\\' . $u[1], '\\'), 1)] = $u[1];
		if ($ext($u[1]) && !$exists($u[1])) {
			$fail("$rel imports $u[1], which does not exist");
		}
	}
	// Foo::CONST / Foo::method( on imported framework classes
	$code = '';
	foreach (token_get_all($src) as $t) {
		$code .= is_array($t) ? (in_array($t[0], [T_COMMENT, T_DOC_COMMENT], true) ? '' : $t[1]) : $t;
	}
	preg_match_all('/\b([A-Z]\w+)::(\w+)(\s*\()?/', $code, $m, PREG_SET_ORDER);
	foreach ($m as $u) {
		$c = $alias[$u[1]] ?? null;
		if (!$c || !$ext($c) || !$exists($c) || $u[2] === 'class') {
			continue;
		}
		$r = new ReflectionClass($c);
		if (!(empty($u[3]) ? $r->hasConstant($u[2]) : $r->hasMethod($u[2]))) {
			$fail("$rel uses $c::$u[2], which does not exist");
		}
	}
	// $this->prop->method() where prop is typed in the constructor
	if (preg_match('/function __construct\((.*?)\)\s*(\{|:)/s', $src, $cm)
		&& preg_match_all('/(?:private|protected|public)\s+(?:readonly\s+)?\??(\w+)\s+\$(\w+)/', $cm[1], $pm, PREG_SET_ORDER)) {
		foreach ($pm as $p) {
			$c = $alias[$p[1]] ?? null;
			if (!$c || !$ext($c) || !$exists($c)) {
				continue;
			}
			preg_match_all('/\$this->' . $p[2] . '->(\w+)\s*\(/', $src, $cm2);
			foreach (array_unique($cm2[1]) as $meth) {
				if (!method_exists($c, $meth)) {
					$fail("$rel calls \$this->$p[2]->$meth() on $c, which does not have it");
				}
			}
		}
	}
	if (preg_match('/@(NoAdminRequired|NoCSRFRequired|PublicPage|AdminRequired|NoTwoFactorRequired)\b/', $src, $am)) {
		$fail("$rel uses the @$am[1] annotation; use the attribute");
	}
	// compile-time check in a child process; files that need the scoped vendor tree are skipped
	if (!str_contains($src, 'Recognize\\Vendor') && preg_match('/^(final |abstract )?(class|interface) /m', $src)) {
		$out = [];
		exec(escapeshellarg(PHP_BINARY) . ' ' . escapeshellarg(__FILE__) . ' ' . escapeshellarg($nc) . ' --load ' . escapeshellarg($f->getPathname()) . ' 2>&1', $out, $rc);
		if ($rc !== 0 || $out) {
			$fail("$rel does not load: " . implode(' ', array_slice($out, 0, 2)));
		}
	}
}
echo $bad === 0 ? "ok: lib/ fits $nc\n" : "$bad problem(s) against $nc\n";
exit($bad === 0 ? 0 : 1);
