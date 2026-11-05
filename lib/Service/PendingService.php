<?php

/*
 * Copyright (c) 2026 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\Service;

use OCA\Recognize\Constants;
use OCP\DB\QueryBuilder\IQueryBuilder;
use OCP\Files\IMimeTypeLoader;
use OCP\IDBConnection;

/**
 * Writes work items to the recognize_pending table.
 *
 * This is the single hand-off point to the Python inference daemon, which
 * polls the table directly; there is no PHP-side classification queue any
 * more. Both the file listener (live events) and the storage crawl job
 * (backfill of pre-existing files) write through here so the upsert and the
 * action values are defined in exactly one place.
 */
class PendingService {
	/** Values are persisted and read by python/db.py — do not renumber. */
	public const ACTION_CREATED = 0;
	public const ACTION_UPDATED = 1;
	public const ACTION_DELETED = 2;

	public function __construct(
		private IDBConnection $db,
		private IMimeTypeLoader $mimeTypes,
	) {
	}

	/**
	 * Upsert a pending item, keyed on file_id.
	 *
	 * Raw SQL because Nextcloud's query builder exposes no ON CONFLICT /
	 * ON DUPLICATE KEY clause.
	 */
	public function upsert(int $fileId, ?int $storageId, int $action): void {
		if ($storageId === null) {
			return;
		}

		$platform = $this->db->getDatabasePlatform();
		$isMySQL = $platform instanceof \Doctrine\DBAL\Platforms\AbstractMySQLPlatform;

		// The connection substitutes *PREFIX*. (The Server getConfig getter this
		// used was removed in Nextcloud 34 and made every upload throw here.)
		$table = '*PREFIX*recognize_pending';

		if ($isMySQL) {
			$sql = "INSERT INTO `{$table}` (`file_id`, `storage_id`, `action`, `added_at`) "
				. 'VALUES (?, ?, ?, ?) '
				. 'ON DUPLICATE KEY UPDATE `action` = VALUES(`action`), `added_at` = VALUES(`added_at`)';
		} else {
			$sql = "INSERT INTO \"{$table}\" (\"file_id\", \"storage_id\", \"action\", \"added_at\") "
				. 'VALUES (?, ?, ?, ?) '
				. 'ON CONFLICT ("file_id") DO UPDATE SET "action" = EXCLUDED."action", "added_at" = EXCLUDED."added_at"';
		}

		$this->db->executeStatement($sql, [$fileId, $storageId, $action, time()]);
	}

	/**
	 * Count pending items grouped by media type.
	 *
	 * recognize_pending holds one row per file, not one per model, so the
	 * per-model figures the admin page shows are derived from the file's
	 * mimetype — the same split the daemon uses to pick classifiers.
	 *
	 * @return array{image:int, video:int, audio:int}
	 */
	public function countByMediaType(): array {
		$groups = [
			'image' => Constants::IMAGE_FORMATS,
			'video' => Constants::VIDEO_FORMATS,
			'audio' => Constants::AUDIO_FORMATS,
		];

		$counts = [];
		foreach ($groups as $group => $formats) {
			$ids = array_values(array_filter(array_map(
				fn ($mimeType) => $this->mimeTypes->getId($mimeType),
				$formats,
			)));
			if ($ids === []) {
				$counts[$group] = 0;
				continue;
			}
			$qb = $this->db->getQueryBuilder();
			$qb->select($qb->func()->count('p.file_id'))
				->from('recognize_pending', 'p')
				->innerJoin('p', 'filecache', 'fc', $qb->expr()->eq('p.file_id', 'fc.fileid'))
				->where($qb->expr()->neq('p.action', $qb->createNamedParameter(self::ACTION_DELETED, IQueryBuilder::PARAM_INT)))
				->andWhere($qb->expr()->in('fc.mimetype', $qb->createNamedParameter($ids, IQueryBuilder::PARAM_INT_ARRAY)));
			$result = $qb->executeQuery();
			$counts[$group] = (int)$result->fetchOne();
			$result->closeCursor();
		}

		return $counts;
	}
}
