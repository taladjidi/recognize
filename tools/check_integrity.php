<?php
/**
 * Drive the real InternalController, StorageCrawlJob, TagManager and face DAV
 * nodes with fakes, and fail if a data-integrity or access-control guard is
 * missing. No database and no PHPUnit needed.
 *
 * WHY: there is no PHPUnit runner on this host, and the faults these guard
 * (cross-user MOVE, partial faces behind a processed tag, a crawl chain that
 * dies on one DB error) only show up as missing data weeks later.
 *
 * The app's final mappers/services are replaced by recording stubs declared
 * before the autoloader can load the real ones. SQL is not exercised: the
 * user_id filters in FaceDetectionMapper need a live instance
 * (tools/probe_propfind.php).
 *
 * Usage: php tools/check_integrity.php [nextcloud_root]
 */
$nc = $argv[1] ?? '/usr/share/nextcloud';
require "$nc/3rdparty/autoload.php";
require "$nc/lib/composer/autoload.php";
require "$nc/apps/dav/composer/autoload.php";
// The real FaceDetectionMapper cannot coexist with its stub; its SQL checks run in a child process.
$realMapper = getenv('CHECK_REAL_MAPPER') === '1';

// Recording stubs for final classes; must exist before the app autoloader runs.
class Stub {
	public array $calls = [];
	public array $on = [];
	public function __call(string $name, array $args): mixed {
		$this->calls[] = [$name, $args];
		Log::$order[] = static::class . '::' . $name;
		if (isset($this->on[$name])) {
			return ($this->on[$name])(...$args);
		}
		return isset($this->on['*']) ? ($this->on['*'])($name, $args) : null;
	}
	public function called(string $name): array {
		return array_values(array_filter($this->calls, fn ($c) => $c[0] === $name));
	}
}
class Log {
	public static array $order = [];
}
if (!$realMapper) {
	eval('namespace OCA\Recognize\Db; class FaceDetectionMapper extends \Stub {} class FaceClusterMapper extends \Stub {}');
	eval('namespace OCA\Recognize\Service; class Logger extends \Stub {} class StorageService extends \Stub {} class PendingService extends \Stub { public const ACTION_CREATED = 0; }');
}

spl_autoload_register(function (string $class): void {
	$prefix = 'OCA\\Recognize\\';
	if (str_starts_with($class, $prefix)) {
		$file = dirname(__DIR__) . '/lib/' . str_replace('\\', '/', substr($class, strlen($prefix))) . '.php';
		if (is_file($file)) {
			require $file;
		}
	}
});

/** Implement an interface with every method routed to Stub::__call. */
function mock(string $iface, array $on = []): object {
	static $n = 0;
	$cls = 'Mock' . (++$n);
	$methods = '';
	$type = static function (?ReflectionType $t) use ($iface): string {
		if ($t === null) {
			return '';
		}
		$s = (string)$t;
		return preg_replace('/\bself\b/', '\\' . $iface, preg_replace('/(?<![\w\\\\])([A-Z]\w*\\\\[\w\\\\]+|[A-Z]\w+)/', '\\\\$1', $s));
	};
	foreach ((new ReflectionClass($iface))->getMethods() as $m) {
		$params = [];
		foreach ($m->getParameters() as $p) {
			$s = $type($p->getType()) . ' ' . ($p->isPassedByReference() ? '&' : '') . ($p->isVariadic() ? '...' : '') . '$' . $p->getName();
			if ($p->isDefaultValueAvailable()) {
				$s .= ' = ' . ($p->isDefaultValueConstant() ? '\\' . preg_replace('/^self::/', $m->getDeclaringClass()->getName() . '::', $p->getDefaultValueConstantName()) : var_export($p->getDefaultValue(), true));
			}
			$params[] = $s;
		}
		$ret = $type($m->getReturnType());
		$body = match ($ret) {
			'void' => '$this->__call(__FUNCTION__, func_get_args());',
			'never' => '$this->__call(__FUNCTION__, func_get_args()); throw new \LogicException("never");',
			default => 'return $this->__call(__FUNCTION__, func_get_args());',
		};
		$methods .= 'public ' . ($m->isStatic() ? 'static ' : '') . 'function ' . $m->getName() . '(' . implode(', ', $params) . ')' . ($ret !== '' ? ": $ret" : '') . " { $body }\n";
	}
	eval("class $cls extends \\Stub implements \\$iface { $methods }");
	$o = new $cls();
	$o->on = $on;
	return $o;
}

$fails = 0;
function check(string $what, bool $ok): void {
	global $fails;
	echo ($ok ? 'ok   ' : 'FAIL ') . $what . "\n";
	$fails += $ok ? 0 : 1;
}
function throws(callable $f, string $class): bool {
	try {
		$f();
	} catch (Throwable $e) {
		return $e instanceof $class;
	}
	return false;
}

use OCA\Recognize\Controller\InternalController;
use OCA\Recognize\Db\FaceDetection;

// --- 0. (child process) the real mapper's queries filter on user_id ---------
if ($realMapper) {
	/** Run $call on a real FaceDetectionMapper over a recording query builder; return the conditions it built. */
	$conditions = function (callable $call): array {
		$expr = mock(\OCP\DB\QueryBuilder\IExpressionBuilder::class, ['*' => fn ($name, $args) => "$name({$args[0]}, {$args[1]})"]);
		$res = mock(\OCP\DB\IResult::class, ['fetch' => fn () => false, 'closeCursor' => fn () => true]);
		$qb = mock(\OCP\DB\QueryBuilder\IQueryBuilder::class);
		$where = [];
		$qb->on = [
			'expr' => fn () => $expr,
			'createNamedParameter' => fn ($v) => 'param:' . json_encode($v),
			'createPositionalParameter' => fn ($v) => 'param:' . json_encode($v),
			'executeQuery' => fn () => $res,
			'*' => function ($name, $args) use ($qb, &$where) {
				if (in_array($name, ['where', 'andWhere'], true)) {
					array_push($where, ...$args);
				}
				return $qb;
			},
		];
		$db = mock(\OCP\IDBConnection::class, ['getQueryBuilder' => fn () => $qb]);
		$call(new \OCA\Recognize\Db\FaceDetectionMapper($db, mock(\OCP\IConfig::class)));
		return $where;
	};
	check('findByFileIdWithTitle filters on the given user', in_array('eq(d.user_id, param:"alice")', $conditions(fn ($m) => $m->findByFileIdWithTitle(9, 'alice')), true));
	check('findByFileIdsWithTitle filters on the given user', in_array('eq(d.user_id, param:"alice")', $conditions(fn ($m) => $m->findByFileIdsWithTitle([9, 10], 'alice')), true));
	exit($fails === 0 ? 0 : 1);
}

// --- 1. validateResults --------------------------------------------------
$face = static fn (array $o = []) => array_merge(['x' => 0.1, 'y' => 0.2, 'width' => 0.3, 'height' => 0.4, 'vector' => array_fill(0, 512, 0.01), 'score' => 0.9], $o);
$v = static fn ($tags, $faces) => InternalController::validateResults($tags, $faces);
check('valid payload accepted', $v(['Cat'], [$face(), $face(['x' => 0, 'width' => 1])]) === null);
check('no faces accepted', $v([], []) === null);
check('box at the edge within eps accepted', $v([], [$face(['x' => 0.7, 'width' => 0.3 + 1e-9])]) === null);
check('int vector components accepted', $v([], [$face(['vector' => array_fill(0, 512, 0)])]) === null);
foreach ([
	'511-dim vector' => [[], [$face(['vector' => array_fill(0, 511, 0.1)])]],
	'513-dim vector' => [[], [$face(['vector' => array_fill(0, 513, 0.1)])]],
	'INF in vector' => [[], [$face(['vector' => array_merge(array_fill(0, 511, 0.1), [INF])])]],
	'NAN in vector' => [[], [$face(['vector' => array_merge([NAN], array_fill(0, 511, 0.1))])]],
	'string in vector' => [[], [$face(['vector' => array_merge(['0.1'], array_fill(0, 511, 0.1))])]],
	'vector as map' => [[], [$face(['vector' => array_combine(range(1, 512), array_fill(0, 512, 0.1))])]],
	'missing vector' => [[], [array_diff_key($face(), ['vector' => 1])]],
	'missing height' => [[], [array_diff_key($face(), ['height' => 1])]],
	'negative x' => [[], [$face(['x' => -0.01])]],
	'width > 1' => [[], [$face(['x' => 0, 'width' => 1.5])]],
	'x + width > 1' => [[], [$face(['x' => 0.8, 'width' => 0.3])]],
	'y + height > 1' => [[], [$face(['y' => 0.8, 'height' => 0.3])]],
	'string coordinate' => [[], [$face(['y' => '0.1'])]],
	'face not an object' => [[], ['nope']],
	'faces as a map' => [[], ['a' => $face()]],
	'faces as a string' => [[], 'x'],
	'tags as a string' => ['Cat', []],
	'int tag' => [[1], []],
	'empty tag' => [[''], []],
] as $what => [$tags, $faces]) {
	check("rejects $what", $v($tags, $faces) !== null);
}

// --- 2. results(): order and transaction --------------------------------
/** @return array{InternalController, Stub, Stub, Stub, Stub, Stub} */
function controller(array $params, array $mapperOn = [], bool $sensitive = true, string $stored = 's3cret', array $tagsOn = []): array {
	Log::$order = [];
	$request = mock(\OCP\IRequest::class, [
		'getParams' => fn () => $params,
		'getHeader' => fn ($h) => $h === 'X-Recognize-Secret' ? 's3cret' : '',
	]);
	$appConfig = mock(\OCP\AppFramework\Services\IAppConfig::class, [
		'getAppValueString' => fn () => $stored,
		'isSensitive' => fn () => $sensitive,
		'setAppValueString' => fn () => true,
	]);
	$tag = mock(\OCP\SystemTag\ISystemTag::class, ['getId' => fn () => '7']);
	$systemTags = mock(\OCP\SystemTag\ISystemTagManager::class, $tagsOn + ['getTag' => fn () => $tag]);
	$objectMapper = mock(\OCP\SystemTag\ISystemTagObjectMapper::class, ['getTagIdsForObjects' => fn ($ids) => [(string)$ids[0] => []]]);
	$mapper = new \OCA\Recognize\Db\FaceDetectionMapper();
	$mapper->on = $mapperOn + ['findByFileIdAndUser' => fn () => []];
	$db = mock(\OCP\IDBConnection::class);
	// writeFaces' owner lookup: a query builder that returns itself and a result yielding 'home::alice'.
	$qb = mock(\OCP\DB\QueryBuilder\IQueryBuilder::class);
	$expr = mock(\OCP\DB\QueryBuilder\IExpressionBuilder::class, ['eq' => fn () => 'a = b']);
	$res = mock(\OCP\DB\IResult::class, ['fetchOne' => fn () => 'home::alice', 'closeCursor' => fn () => true]);
	$qb->on = ['select' => fn () => $qb, 'from' => fn () => $qb, 'innerJoin' => fn () => $qb, 'where' => fn () => $qb,
		'expr' => fn () => $expr, 'createNamedParameter' => fn () => ':p', 'executeQuery' => fn () => $res];
	$db->on['getQueryBuilder'] = fn () => $qb;
	$alice = mock(\OCP\IUser::class, ['getUID' => fn () => 'alice']);
	$users = mock(\OCP\IUserManager::class, ['get' => fn ($uid) => $uid === 'alice' ? $alice : null]);
	$c = new InternalController('recognize', $request, new \OCA\Recognize\Service\TagManager($systemTags, $objectMapper),
		$mapper, $appConfig, mock(\Psr\Log\LoggerInterface::class), $db, $users);
	return [$c, $mapper, $db, $objectMapper, $appConfig, $systemTags];
}

// NC 34 refuses createTag() without a user outside the CLI: a new label must be created as the file's owner.
[$c, , , $objects, , $sys] = controller(['file_id' => 42, 'tags' => ['NewLabel'], 'faces' => []], [], true, 's3cret', [
	'getTag' => function () { throw new \OCP\SystemTag\TagNotFoundException(); },
	'createTag' => fn ($name, $visible, $assignable, $user = null) => mock(\OCP\SystemTag\ISystemTag::class, ['getId' => fn () => '9']),
]);
$r = $c->results();
$created = $sys->called('createTag');
check('new label -> created as the file owner', $r->getStatus() === 200 && $created !== []
	&& array_reduce($created, fn ($ok, $call) => $ok && ($call[1][3] ?? null) !== null && $call[1][3]->getUID() === 'alice', true));

[$c, $mapper, $db, $objects] = controller(['file_id' => 42, 'tags' => ['Cat'], 'faces' => [$face(['vector' => array_fill(0, 3, 0.1)])]]);
$r = $c->results();
check('invalid face -> 400', $r->getStatus() === 400);
check('invalid face -> nothing written, no processed tag', $mapper->called('insert') === [] && $db->called('beginTransaction') === [] && $objects->called('assignTags') === []);

$n = 0;
[$c, $mapper, $db, $objects] = controller(['file_id' => 42, 'tags' => ['Cat'], 'faces' => [$face(), $face(['x' => 0.5])]],
	['insert' => function ($d) use (&$n) {
		if (++$n === 2) {
			throw new \OCP\DB\Exception('disk full');
		}
		return $d;
	}]);
$r = $c->results();
check('insert fails mid-way -> 500 on the second insert', $r->getStatus() === 500 && $n === 2 && str_contains($r->getData()['error'], 'disk full'));
check('insert fails mid-way -> rolled back, not committed', count($db->called('rollBack')) === 1 && $db->called('commit') === []);
check('insert fails mid-way -> file not tagged processed (will be retried)', $objects->called('assignTags') === []);

$nextId = 100;
[$c, $mapper, $db, $objects] = controller(['file_id' => 42, 'tags' => ['Cat'], 'faces' => [$face(), $face(['x' => 0.5])]],
	['insert' => function (FaceDetection $d) use (&$nextId) {
		$d->setId($nextId++);
		return $d;
	}]);
$r = $c->results();
$order = array_values(array_filter(Log::$order, fn ($s) => preg_match('/beginTransaction|::insert$|commit|::assignTags/', $s)));
check('success -> 200 ' . json_encode($r->getData()), $r->getStatus() === 200);
check('success -> begin, insert x2, commit, then tags', array_map(fn ($s) => preg_replace('/^.*::/', '', $s), $order) === ['beginTransaction', 'insert', 'insert', 'commit', 'assignTags']);
$ins = $mapper->called('insert');
check('success -> owner and coordinates stored', $ins[0][1][0]->getUserId() === 'alice' && $ins[1][1][0]->getX() === 0.5 && count($ins[0][1][0]->getVector()) === 512);
check('success -> detection ids returned in input order', $r->getData() === ['status' => 'ok', 'detection_ids' => [100, 101]]);
check('success -> processed tag assigned with label', in_array('7', $objects->called('assignTags')[0][1][2], true));

[$c, $mapper, $db, $objects] = controller(['file_id' => 42, 'tags' => [], 'faces' => [$face()]],
	['findByFileIdAndUser' => fn () => [new FaceDetection()]]);
$r = $c->results();
check('existing faces kept -> no insert, still tagged', $r->getStatus() === 200 && $mapper->called('insert') === [] && count($objects->called('assignTags')) === 1);
check('existing faces kept -> detection_ids is []', $r->getData()['detection_ids'] === []);

[$c, , , , $appConfig] = controller(['file_id' => 42], [], false);
$c->results();
$set = $appConfig->called('setAppValueString');
check('plain stored secret re-saved as sensitive', count($set) === 1 && $set[0][1] === ['internal_secret', 's3cret', false, true]);
[$c, , , , $appConfig] = controller(['file_id' => 42], [], true);
$c->results();
check('sensitive secret not rewritten', $appConfig->called('setAppValueString') === []);
[$c, , , , $appConfig] = controller(['file_id' => 42], [], false, '');
check('no secret -> generated as sensitive, request refused', $c->results()->getStatus() === 401 && ($appConfig->called('setAppValueString')[0][1][3] ?? null) === true);
check('file endpoint removed', !method_exists(InternalController::class, 'file'));
check('file route removed', !str_contains(file_get_contents(dirname(__DIR__) . '/appinfo/routes.php'), 'internal#file'));

// --- 3. StorageCrawlJob re-adds itself on failure --------------------------
function crawl(callable $files, callable $upsert): ?array {
	$jobList = mock(\OCP\BackgroundJob\IJobList::class);
	$storage = new \OCA\Recognize\Service\StorageService();
	$storage->on['getFilesInMount'] = $files;
	$pending = new \OCA\Recognize\Service\PendingService();
	$pending->on['upsert'] = $upsert;
	$job = new \OCA\Recognize\BackgroundJobs\StorageCrawlJob(mock(\OCP\AppFramework\Utility\ITimeFactory::class),
		new \OCA\Recognize\Service\Logger(), $jobList, $storage, $pending,
		new \OCA\Recognize\Service\TagManager(mock(\OCP\SystemTag\ISystemTagManager::class), mock(\OCP\SystemTag\ISystemTagObjectMapper::class)));
	$run = new ReflectionMethod($job, 'run');
	try {
		$run->invoke($job, ['storage_id' => 3, 'root_id' => 4, 'override_root' => 5, 'last_file_id' => 100, 'models' => ['faces'], 'skip_tagged' => false]);
	} catch (\OCP\DB\Exception $e) {
		// Escaping the job is the old failure: its row is already removed, so the chain ends.
	}
	$added = $jobList->called('add');
	return $added === [] ? null : $added[0][1][1];
}
$gen = function () {
	yield ['fileid' => 101];
	yield ['fileid' => 102];
	yield ['fileid' => 103];
};
$a = crawl($gen, fn ($id) => $id === 103 ? throw new \OCP\DB\Exception('deadlock') : null);
check('upsert fails -> job re-added from the last queued file', ($a['last_file_id'] ?? null) === 102 && $a['models'] === ['faces'] && $a['skip_tagged'] === false && $a['storage_id'] === 3);
$a = crawl(function () {
	throw new \OCP\DB\Exception('gone');
	yield;
}, fn () => null);
check('file query fails -> job re-added with the same last_file_id', ($a['last_file_id'] ?? null) === 100);
$a = crawl($gen, fn () => null);
check('batch ok -> next iteration scheduled', ($a['last_file_id'] ?? null) === 103);
check('storage done -> chain ends', crawl(fn () => yield from [], fn () => null) === null);

// --- 4. TagManager::removeEmptyTags spares other tags -----------------------
$tags = [];
foreach (['Cat', 'Vacation'] as $i => $name) {
	$tags[$name] = mock(\OCP\SystemTag\ISystemTag::class, ['getId' => fn () => (string)$i, 'getName' => fn () => $name]);
}
$systemTags = mock(\OCP\SystemTag\ISystemTagManager::class, [
	'getAllTags' => fn () => array_values($tags),
	'getTag' => fn ($name) => $tags[$name] ?? throw new \OCP\SystemTag\TagNotFoundException(),
]);
(new \OCA\Recognize\Service\TagManager($systemTags, mock(\OCP\SystemTag\ISystemTagObjectMapper::class, ['getObjectIdsForTags' => fn () => []])))->removeEmptyTags();
check('removeEmptyTags deletes the empty classifier tag and not the user tag', array_map(fn ($c) => $c[1][0], $systemTags->called('deleteTags')) === ['0']);

// --- 5. DAV nodes: ownership and guards --------------------------------------
$alice = mock(\OCP\IUser::class, ['getUID' => fn () => 'alice']);
$rootFolder = mock(\OCP\Files\IRootFolder::class, ['getUserFolder' => fn () => mock(\OCP\Files\Folder::class)]);
$det = static function (int $id, string $user, ?int $cluster): FaceDetection {
	$d = new FaceDetection();
	$d->setId($id);
	$d->setUserId($user);
	$d->setClusterId($cluster);
	$d->setFileId(9);
	$d->setVector(array_fill(0, 512, 0.0));
	return $d;
};
$dm = new \OCA\Recognize\Db\FaceDetectionMapper();
$dm->on = [
	'find' => fn ($id) => $det($id, $id === 5 ? 'bob' : 'alice', -1),
	'findRejectedByUserIdAndFileName' => fn () => throw new \OCP\AppFramework\Db\DoesNotExistException(''),
];
$tm = mock(\OCP\ITagManager::class);
$pv = mock(\OCP\IPreview::class);
$home = new \OCA\Recognize\Dav\Faces\UnassignedFacesHome($alice, $dm, $rootFolder, $tm, $pv);
check("unassigned: another user's detection is not found", throws(fn () => $home->getChild('5-x.jpg'), \Sabre\DAV\Exception\NotFound::class));
check('unassigned: own detection is found', $home->getChild('6-x.jpg')->getFaceDetection()->getId() === 6);

$cluster = new \OCA\Recognize\Db\FaceCluster();
$cluster->setId(77);
$cluster->setUserId('alice');
$cm = new \OCA\Recognize\Db\FaceClusterMapper();
$cm->on['findByUserAndTitle'] = fn () => throw new \OCP\AppFramework\Db\DoesNotExistException('');
$root = new \OCA\Recognize\Dav\Faces\FaceRoot($cm, $cluster, $alice, $dm, $rootFolder, $tm, $pv);
$bobs = new \OCA\Recognize\Dav\Faces\FacePhoto($dm, $det(5, 'bob', -1), mock(\OCP\Files\Folder::class), $tm, $pv);
check("moveInto: another user's detection refused", throws(fn () => $root->moveInto('x', 'y', $bobs), \Sabre\DAV\Exception\Forbidden::class) && $dm->called('update') === []);
$mine = new \OCA\Recognize\Dav\Faces\FacePhoto($dm, $det(6, 'alice', -1), mock(\OCP\Files\Folder::class), $tm, $pv);
check('moveInto: own detection moved into the cluster', $root->moveInto('x', 'y', $mine) === true && $mine->getFaceDetection()->getClusterId() === 77);
check('setName: numeric title refused', throws(fn () => $root->setName('123'), \Sabre\DAV\Exception\Forbidden::class) && $cm->called('update') === []);
$root->setName('Alice');
check('setName: normal title saved', $cluster->getTitle() === 'Alice' && count($cm->called('update')) === 1);

$orphan = new \OCA\Recognize\Dav\Faces\FacePhoto($dm, $det(8, 'alice', null), mock(\OCP\Files\Folder::class), $tm, $pv);
check('FacePhoto::delete with no cluster does not throw', !throws(fn () => $orphan->delete(), Throwable::class) && $dm->called('findByClusterId') === []);

// --- 6. face-detections on a plain file: only the session user's ------------
$propFindFor = function (?\OCP\IUser $sessionUser, \OCA\Recognize\Db\FaceDetectionMapper $dm): mixed {
	$plugin = new \OCA\Recognize\Dav\Faces\PropFindPlugin($dm, mock(\OCP\IPreview::class), new \OCA\Recognize\Db\FaceClusterMapper(),
		mock(\OCP\IUserSession::class, ['getUser' => fn () => $sessionUser]), mock(\OCP\Security\ICrypto::class),
		mock(\Psr\Log\LoggerInterface::class), mock(\OCP\AppFramework\Utility\ITimeFactory::class));
	$file = new class extends \OCA\DAV\Connector\Sabre\File {
		public function __construct() {
		}
		public function getId(): ?int {
			return 42;
		}
	};
	$propFind = new \Sabre\DAV\PropFind('files/bob/x.jpg', [\OCA\Recognize\Dav\Faces\PropFindPlugin::FACE_DETECTIONS_PROPERTYNAME]);
	$plugin->propFind($propFind, $file);
	return $propFind->get(\OCA\Recognize\Dav\Faces\PropFindPlugin::FACE_DETECTIONS_PROPERTYNAME);
};
$dm = new \OCA\Recognize\Db\FaceDetectionMapper();
$dm->on['findByFileIdWithTitle'] = fn () => [];
$propFindFor($alice, $dm);
$asked = $dm->called('findByFileIdWithTitle');
check("file face-detections: asks for the session user's detections only", count($asked) === 1 && $asked[0][1] === [42, 'alice']);
$dm = new \OCA\Recognize\Db\FaceDetectionMapper();
$dm->on['findByFileIdWithTitle'] = fn () => [];
check('file face-detections: no session -> [] without a query', $propFindFor(null, $dm) === '[]' && $dm->calls === []);

// --- 7. face owner from the storage id --------------------------------------
check('owner: local home storage', InternalController::ownerFromStorageId('home::alice') === 'alice');
check('owner: S3 primary storage', InternalController::ownerFromStorageId('object::user:alice') === 'alice');
check('owner: no single owner -> empty', InternalController::ownerFromStorageId('local::/mnt/photos/') === '' && InternalController::ownerFromStorageId('') === '');

// --- 8. CatchUpJob crawls only enabled models -------------------------------
function catchUp(array $config): array {
	$jobList = mock(\OCP\BackgroundJob\IJobList::class, ['getJobsIterator' => fn () => []]);
	$storage = new \OCA\Recognize\Service\StorageService();
	$storage->on['getMounts'] = fn () => [['storage_id' => 3, 'root_id' => 4, 'override_root' => 4]];
	$appConfig = mock(\OCP\AppFramework\Services\IAppConfig::class, [
		'hasAppKey' => fn ($k) => isset($config[$k]),
		'getAppValueString' => fn ($k, $d = '') => $config[$k] ?? $d,
	]);
	$job = new \OCA\Recognize\BackgroundJobs\CatchUpJob(mock(\OCP\AppFramework\Utility\ITimeFactory::class), $jobList, $storage,
		mock(\Psr\Log\LoggerInterface::class), $appConfig);
	(new ReflectionMethod($job, 'run'))->invoke($job, null);
	return array_map(fn ($c) => $c[1][1]['models'] ?? \OCA\Recognize\Constants::MODELS, $jobList->called('add'));
}
check('catch-up: only enabled models crawled', catchUp(['faces.enabled' => 'true', 'imagenet.enabled' => 'true', 'movinet.enabled' => 'false', 'musicnn.enabled' => 'false']) === [['faces', 'imagenet']]);
check('catch-up: nothing enabled -> no crawl', catchUp(['faces.enabled' => 'false', 'movinet.enabled' => 'false']) === []);
check('catch-up: no settings yet -> all models, like the daemon', catchUp([]) === [\OCA\Recognize\Constants::MODELS]);

// --- 0, in a child process with the real mapper ------------------------------
exec('CHECK_REAL_MAPPER=1 ' . escapeshellarg(PHP_BINARY) . ' ' . escapeshellarg(__FILE__) . ' ' . escapeshellarg($nc) . ' 2>&1', $out, $code);
echo implode("\n", $out), "\n";
$fails += substr_count(implode("\n", $out), 'FAIL ') + ($code !== 0 && !str_contains(implode("\n", $out), 'FAIL ') ? 1 : 0);

echo $fails === 0 ? "all checks passed\n" : "$fails check(s) failed\n";
exit($fails === 0 ? 0 : 1);
