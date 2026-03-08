<?php

/*
 * Copyright (c) 2022-2026 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\Hooks;

use OCA\Recognize\Constants;
use OCA\Recognize\Service\IgnoreService;
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
 * Actions: 0=created, 1=updated, 2=deleted.
 *
 * @template-implements IEventListener<Event>
 */
final class FileListener implements IEventListener {
	public function __construct(
		private LoggerInterface $logger,
		private IgnoreService $ignoreService,
		private IDBConnection $db,
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
				$this->upsertPending($node->getId(), $node->getMountPoint()->getNumericStorageId(), 0);
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
					$this->upsertPending($target->getId(), $target->getMountPoint()->getNumericStorageId(), 2);
				} else {
					// Moved to non-ignored location — treat as update (re-classify)
					$this->upsertPending($target->getId(), $target->getMountPoint()->getNumericStorageId(), 1);
				}
			}

			if ($event instanceof BeforeNodeDeletedEvent) {
				$node = $event->getNode();
				if ($node->getType() === FileInfo::TYPE_FOLDER) {
					$this->deleteFolder($node);
					return;
				}
				$this->upsertPending($node->getId(), $node->getMountPoint()->getNumericStorageId(), 2);
			}
		} catch (\Throwable $e) {
			$this->logger->error('Error in recognize file listener', ['exception' => $e]);
		}
	}

	/**
	 * Upsert into recognize_pending.
	 *
	 * Uses raw SQL for the ON CONFLICT clause which isn't available in
	 * Nextcloud's query builder.
	 */
	private function upsertPending(int $fileId, ?int $storageId, int $action): void {
		if ($storageId === null) {
			return;
		}
		$now = time();

		$platform = $this->db->getDatabasePlatform();
		$isMySQL = $platform instanceof \Doctrine\DBAL\Platforms\AbstractMySQLPlatform;

		$prefix = \OC::$server->getConfig()->getSystemValueString('dbtableprefix', 'oc_');
		$table = $prefix . 'recognize_pending';

		if ($isMySQL) {
			$sql = "INSERT INTO `{$table}` (`file_id`, `storage_id`, `action`, `added_at`) "
				. "VALUES (?, ?, ?, ?) "
				. "ON DUPLICATE KEY UPDATE `action` = VALUES(`action`), `added_at` = VALUES(`added_at`)";
		} else {
			$sql = "INSERT INTO \"{$table}\" (\"file_id\", \"storage_id\", \"action\", \"added_at\") "
				. "VALUES (?, ?, ?, ?) "
				. "ON CONFLICT (\"file_id\") DO UPDATE SET \"action\" = EXCLUDED.\"action\", \"added_at\" = EXCLUDED.\"added_at\"";
		}

		$this->db->executeStatement($sql, [$fileId, $storageId, $action, $now]);
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
					$this->upsertPending($child->getId(), $child->getMountPoint()->getNumericStorageId(), 2);
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
