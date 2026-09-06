<?php

/*
 * Copyright (c) 2022-2026 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\Hooks;

use OCA\Recognize\Constants;
use OCA\Recognize\Service\IgnoreService;
use OCA\Recognize\Service\PendingService;
use OCP\DB\Exception;
use OCP\DB\QueryBuilder\IQueryBuilder;
use OCP\EventDispatcher\Event;
use OCP\EventDispatcher\IEventListener;
use OCP\Files\Events\Node\BeforeNodeDeletedEvent;
use OCP\Files\Events\Node\NodeCreatedEvent;
use OCP\Files\Events\Node\NodeRenamedEvent;
use OCP\Files\FileInfo;
use OCP\Files\Folder;
use OCP\Files\Node;
use OCP\Files\NotFoundException;
use OCP\IDBConnection;
use Psr\Log\LoggerInterface;

/**
 * Simplified file listener: upserts into recognize_pending table.
 *
 * The Python daemon polls this table directly — no background jobs needed.
 *
 * @template-implements IEventListener<Event>
 */
final class FileListener implements IEventListener {
	public function __construct(
		private LoggerInterface $logger,
		private IgnoreService $ignoreService,
		private IDBConnection $db,
		private PendingService $pending,
	) {
	}

	public function handle(Event $event): void {
		try {
			if ($event instanceof NodeCreatedEvent) {
				$node = $event->getNode();
				if ($node->getType() === FileInfo::TYPE_FOLDER) {
					return;
				}
				if (!$this->isSupportedMimetype($node->getMimetype())) {
					return;
				}
				if ($this->isFileIgnored($node)) {
					return;
				}
				$this->pending->upsert($node->getId(), $node->getMountPoint()->getNumericStorageId(), PendingService::ACTION_CREATED);
			}

			if ($event instanceof NodeRenamedEvent) {
				$target = $event->getTarget();
				if ($target->getType() === FileInfo::TYPE_FOLDER) {
					return;
				}
				if (!$this->isSupportedMimetype($target->getMimetype())) {
					return;
				}
				if ($this->isFileIgnored($target)) {
					// Moved into ignored territory — treat as deletion
					$this->pending->upsert($target->getId(), $target->getMountPoint()->getNumericStorageId(), PendingService::ACTION_DELETED);
				} else {
					// Moved to non-ignored location — treat as update (re-classify)
					$this->pending->upsert($target->getId(), $target->getMountPoint()->getNumericStorageId(), PendingService::ACTION_UPDATED);
				}
			}

			if ($event instanceof BeforeNodeDeletedEvent) {
				$node = $event->getNode();
				if ($node->getType() === FileInfo::TYPE_FOLDER) {
					$this->deleteFolder($node);
					return;
				}
				$this->pending->upsert($node->getId(), $node->getMountPoint()->getNumericStorageId(), PendingService::ACTION_DELETED);
			}
		} catch (\Throwable $e) {
			$this->logger->error('Error in recognize file listener', ['exception' => $e]);
		}
	}


	/**
	 * Recursively insert deletion entries for all files in a folder.
	 */
	private function deleteFolder(Node $node): void {
		try {
			/** @var Folder $node */
			foreach ($node->getDirectoryListing() as $child) {
				if ($child->getType() === FileInfo::TYPE_FOLDER) {
					$this->deleteFolder($child);
				} else {
					$this->pending->upsert($child->getId(), $child->getMountPoint()->getNumericStorageId(), PendingService::ACTION_DELETED);
				}
			}
		} catch (NotFoundException $e) {
			$this->logger->debug($e->getMessage(), ['exception' => $e]);
		}
	}

	private function isSupportedMimetype(string $mimetype): bool {
		return in_array($mimetype, Constants::IMAGE_FORMATS, true)
			|| in_array($mimetype, Constants::VIDEO_FORMATS, true)
			|| in_array($mimetype, Constants::AUDIO_FORMATS, true);
	}

	private function isFileIgnored(Node $node): bool {
		$storageId = $node->getMountPoint()->getNumericStorageId();
		if ($storageId === null) {
			return true;
		}

		// Only queue files under user/files/ or groupfolders/
		if (preg_match('#^/[^/]*?/files($|/)#', $node->getPath()) !== 1
			&& preg_match('#^/groupfolders/#', $node->getPath()) !== 1) {
			return true;
		}

		$ignoreMarkers = Constants::IGNORE_MARKERS_ALL;
		$mimeType = $node->getMimetype();

		if (in_array($mimeType, Constants::IMAGE_FORMATS, true)) {
			$ignoreMarkers = array_merge($ignoreMarkers, Constants::IGNORE_MARKERS_IMAGE);
		}
		if (in_array($mimeType, Constants::VIDEO_FORMATS, true)) {
			$ignoreMarkers = array_merge($ignoreMarkers, Constants::IGNORE_MARKERS_VIDEO);
		}
		if (in_array($mimeType, Constants::AUDIO_FORMATS, true)) {
			$ignoreMarkers = array_merge($ignoreMarkers, Constants::IGNORE_MARKERS_AUDIO);
		}

		$ignoredPaths = $this->ignoreService->getIgnoredDirectories($storageId, $ignoreMarkers);

		foreach ($ignoredPaths as $ignoredPath) {
			if (stripos($node->getInternalPath(), $ignoredPath ? $ignoredPath . '/' : $ignoredPath) === 0) {
				return true;
			}
		}

		return false;
	}
}
