<?php

declare(strict_types=1);

/*
 * Copyright (c) 2024-2026 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */

namespace OCA\Recognize\Controller;

use OCA\Recognize\Db\FaceDetection;
use OCA\Recognize\Db\FaceDetectionMapper;
use OCA\Recognize\Service\TagManager;
use OCP\AppFramework\Controller;
use OCP\AppFramework\Http;
use OCP\AppFramework\Http\JSONResponse;
use OCP\AppFramework\Services\IAppConfig;
use OCP\IDBConnection;
use OCP\IRequest;
use Psr\Log\LoggerInterface;

/**
 * Internal API for the Python inference daemon.
 *
 * Endpoints:
 *   POST /api/internal/results  — Write tags and/or face detections for a file
 *
 * Authentication: X-Recognize-Secret header, validated against appconfig.
 */
final class InternalController extends Controller {
	public function __construct(
		string $appName,
		IRequest $request,
		private TagManager $tagManager,
		private FaceDetectionMapper $faceDetectionMapper,
		private IAppConfig $appConfig,
		private LoggerInterface $logger,
		private IDBConnection $db,
	) {
		parent::__construct($appName, $request);
	}

	public const VECTOR_DIM = 512;
	private const BBOX_EPSILON = 1e-6;

	/**
	 * @NoCSRFRequired
	 * @PublicPage
	 * @NoAdminRequired
	 */
	public function results(): JSONResponse {
		if (!$this->validateSecret()) {
			return new JSONResponse(['error' => 'Unauthorized'], Http::STATUS_UNAUTHORIZED);
		}

		$body = $this->request->getParams();
		$fileId = (int)($body['file_id'] ?? 0);
		if ($fileId <= 0) {
			return new JSONResponse(['error' => 'Missing file_id'], Http::STATUS_BAD_REQUEST);
		}

		$tags = $body['tags'] ?? [];
		$faces = $body['faces'] ?? [];
		$error = self::validateResults($tags, $faces);
		if ($error !== null) {
			return new JSONResponse(['error' => $error], Http::STATUS_BAD_REQUEST);
		}
		/** @var list<string> $tags */
		/** @var list<array{x: float|int, y: float|int, width: float|int, height: float|int, vector: list<float|int>}> $faces */

		// Faces first, tags last: assignTags() adds the processed tag, and a
		// processed file is never retried, so its faces must be stored first.
		if (count($faces) > 0) {
			try {
				$this->writeFaces($fileId, $faces);
			} catch (\Throwable $e) {
				$this->logger->error('Failed to write faces for file ' . $fileId, ['exception' => $e]);
				return new JSONResponse(
					['error' => 'Face write failed: ' . $e->getMessage()],
					Http::STATUS_INTERNAL_SERVER_ERROR
				);
			}
		}

		// Always, even with no labels: a file that never gets the processed tag
		// is re-queued by CatchUpJob every day.
		try {
			$this->tagManager->assignTags($fileId, $tags);
		} catch (\Throwable $e) {
			$this->logger->error('Failed to assign tags for file ' . $fileId, ['exception' => $e]);
			return new JSONResponse(
				['error' => 'Tag assignment failed: ' . $e->getMessage()],
				Http::STATUS_INTERNAL_SERVER_ERROR
			);
		}

		return new JSONResponse(['status' => 'ok']);
	}

	/**
	 * Check the daemon's payload before anything is written.
	 *
	 * @return ?string the error, or null when the payload is valid
	 */
	public static function validateResults(mixed $tags, mixed $faces): ?string {
		if (!is_array($tags) || !array_is_list($tags)) {
			return 'tags must be a list';
		}
		foreach ($tags as $tag) {
			if (!is_string($tag) || $tag === '') {
				return 'tags must be non-empty strings';
			}
		}
		if (!is_array($faces) || !array_is_list($faces)) {
			return 'faces must be a list';
		}
		$isNumber = static fn (mixed $v): bool => (is_int($v) || is_float($v)) && is_finite((float)$v);
		foreach ($faces as $i => $face) {
			if (!is_array($face)) {
				return "face $i is not an object";
			}
			$vector = $face['vector'] ?? null;
			if (!is_array($vector) || !array_is_list($vector) || count($vector) !== self::VECTOR_DIM) {
				return "face $i: vector must be a list of " . self::VECTOR_DIM . ' numbers';
			}
			foreach ($vector as $v) {
				if (!$isNumber($v)) {
					return "face $i: vector must contain only finite numbers";
				}
			}
			foreach (['x', 'y', 'width', 'height'] as $key) {
				$v = $face[$key] ?? null;
				if (!$isNumber($v) || $v < 0 || $v > 1) {
					return "face $i: $key must be a number in 0..1";
				}
			}
			if ($face['x'] + $face['width'] > 1 + self::BBOX_EPSILON || $face['y'] + $face['height'] > 1 + self::BBOX_EPSILON) {
				return "face $i: box extends past the image";
			}
		}
		return null;
	}

	/**
	 * Validate the X-Recognize-Secret header against stored secret.
	 */
	private function validateSecret(): bool {
		$stored = $this->appConfig->getAppValueString('internal_secret', '');
		if ($stored === '') {
			// Auto-generate secret on first call
			$stored = bin2hex(random_bytes(32));
			$this->appConfig->setAppValueString('internal_secret', $stored, false, true);
			$this->logger->info('Generated new internal secret for recognize');
		} elseif (!$this->appConfig->isSensitive('internal_secret')) {
			// Stored before it was marked sensitive (or by an older occ call): encrypt it in place, once.
			$this->appConfig->setAppValueString('internal_secret', $stored, false, true);
		}

		$header = $this->request->getHeader('X-Recognize-Secret');
		if ($header === '') {
			return false;
		}

		return hash_equals($stored, $header);
	}

	/**
	 * Write face detections to DB.
	 *
	 * @param int $fileId
	 * @param array $faces Each face: {x, y, width, height, vector, score, user_id}
	 */
	/**
	 * The owning user of a home storage: local (`home::<uid>`) or S3 primary (`object::user:<uid>`).
	 *
	 * shortcut: group folders and external storage have no single owner and get ''; their faces show in no view.
	 */
	public static function ownerFromStorageId(string $storageId): string {
		foreach (['home::', 'object::user:'] as $prefix) {
			if (str_starts_with($storageId, $prefix)) {
				return substr($storageId, strlen($prefix));
			}
		}
		return '';
	}

	private function writeFaces(int $fileId, array $faces): void {
		// All or nothing: a partial set would be kept forever by the early return below.
		$this->db->beginTransaction();
		try {
			$this->insertFaces($fileId, $faces);
			$this->db->commit();
		} catch (\Throwable $e) {
			$this->db->rollBack();
			throw $e;
		}
	}

	private function insertFaces(int $fileId, array $faces): void {
		// Resolve file owner from storage ID (works without user session)
		$qb = $this->db->getQueryBuilder();
		$result = $qb->select('s.id')
			->from('filecache', 'f')
			->innerJoin('f', 'storages', 's', $qb->expr()->eq('f.storage', 's.numeric_id'))
			->where($qb->expr()->eq('f.fileid', $qb->createNamedParameter($fileId)))
			->executeQuery();
		$storageId = $result->fetchOne();
		$result->closeCursor();
		$userId = self::ownerFromStorageId(is_string($storageId) ? $storageId : '');

		// A rename or move re-queues the file; keep the faces it already has
		// (and their cluster assignments) instead of inserting a second set.
		// insert()'s own de-duplication binds the float coordinates as ints,
		// so it only ever matches faces at exactly x = 0.
		// shortcut: content replaced in place keeps its old faces; NodeWrittenEvent is not listened to anyway.
		if ($this->faceDetectionMapper->findByFileIdAndUser($fileId, $userId) !== []) {
			return;
		}

		foreach ($faces as $face) {
			$detection = new FaceDetection();
			$detection->setFileId($fileId);
			$detection->setUserId($userId);
			$detection->setX((float)$face['x']);
			$detection->setY((float)$face['y']);
			$detection->setWidth((float)$face['width']);
			$detection->setHeight((float)$face['height']);
			$detection->setVector(array_map('floatval', $face['vector']));
			$this->faceDetectionMapper->insert($detection);
		}
	}
}
