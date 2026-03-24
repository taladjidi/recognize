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
use OCP\AppFramework\Http\DataResponse;
use OCP\AppFramework\Http\JSONResponse;
use OCP\AppFramework\Http\StreamResponse;
use OCP\AppFramework\Services\IAppConfig;
use OCP\Files\IRootFolder;
use OCP\Files\NotFoundException;
use OCP\IRequest;
use Psr\Log\LoggerInterface;

/**
 * Internal API for the Python inference daemon.
 *
 * Endpoints:
 *   POST /api/internal/results  — Write tags and/or face detections for a file
 *   GET  /api/internal/file/{id} — Stream file content (for S3/encrypted storage)
 *
 * Authentication: X-Recognize-Secret header, validated against appconfig.
 */
final class InternalController extends Controller {
	public function __construct(
		string $appName,
		IRequest $request,
		private TagManager $tagManager,
		private FaceDetectionMapper $faceDetectionMapper,
		private IRootFolder $rootFolder,
		private IAppConfig $appConfig,
		private LoggerInterface $logger,
	) {
		parent::__construct($appName, $request);
	}

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

		// Write tags via ISystemTagManager (fires events, updates etags)
		$tags = $body['tags'] ?? null;
		if (is_array($tags) && count($tags) > 0) {
			try {
				$this->tagManager->assignTags($fileId, $tags);
			} catch (\Throwable $e) {
				$this->logger->error('Failed to assign tags for file ' . $fileId, ['exception' => $e]);
				return new JSONResponse(
					['error' => 'Tag assignment failed: ' . $e->getMessage()],
					Http::STATUS_INTERNAL_SERVER_ERROR
				);
			}
		}

		// Write face detections directly to DB
		$faces = $body['faces'] ?? null;
		if (is_array($faces) && count($faces) > 0) {
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

		return new JSONResponse(['status' => 'ok']);
	}

	/**
	 * @NoCSRFRequired
	 * @PublicPage
	 * @NoAdminRequired
	 */
	public function file(int $id): Http\Response {
		if (!$this->validateSecret()) {
			return new JSONResponse(['error' => 'Unauthorized'], Http::STATUS_UNAUTHORIZED);
		}

		try {
			$nodes = $this->rootFolder->getById($id);
			if (empty($nodes)) {
				return new JSONResponse(['error' => 'File not found'], Http::STATUS_NOT_FOUND);
			}
			$node = $nodes[0];
			if ($node->getType() !== \OCP\Files\FileInfo::TYPE_FILE) {
				return new JSONResponse(['error' => 'Not a file'], Http::STATUS_BAD_REQUEST);
			}

			/** @var \OCP\Files\File $node */
			$response = new DataResponse('');
			$response->setStatus(Http::STATUS_OK);

			// Use a callback response for streaming
			$content = $node->getContent();
			$mimeType = $node->getMimetype();

			$response = new \OCP\AppFramework\Http\DataDownloadResponse(
				$content,
				$node->getName(),
				$mimeType
			);
			return $response;
		} catch (NotFoundException $e) {
			return new JSONResponse(['error' => 'File not found'], Http::STATUS_NOT_FOUND);
		} catch (\Throwable $e) {
			$this->logger->error('Failed to stream file ' . $id, ['exception' => $e]);
			return new JSONResponse(
				['error' => 'File read failed'],
				Http::STATUS_INTERNAL_SERVER_ERROR
			);
		}
	}

	/**
	 * Validate the X-Recognize-Secret header against stored secret.
	 */
	private function validateSecret(): bool {
		$stored = $this->appConfig->getAppValueString('internal_secret', '');
		if ($stored === '') {
			// Auto-generate secret on first call
			$stored = bin2hex(random_bytes(32));
			$this->appConfig->setAppValueString('internal_secret', $stored);
			$this->logger->info('Generated new internal secret for recognize');
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
	private function writeFaces(int $fileId, array $faces): void {
		foreach ($faces as $face) {
			$detection = new FaceDetection();
			$detection->setFileId($fileId);
			$detection->setUserId($face['user_id'] ?? '');
			$detection->setX((float)($face['x'] ?? 0));
			$detection->setY((float)($face['y'] ?? 0));
			$detection->setWidth((float)($face['width'] ?? 0));
			$detection->setHeight((float)($face['height'] ?? 0));
			$detection->setVector($face['vector'] ?? []);
			$this->faceDetectionMapper->insert($detection);
		}
	}
}
