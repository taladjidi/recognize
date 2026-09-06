<?php

/*
 * Copyright (c) 2021-2026 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\BackgroundJobs;

use OCA\Recognize\Constants;
use OCA\Recognize\Service\Logger;
use OCA\Recognize\Service\PendingService;
use OCA\Recognize\Service\StorageService;
use OCP\AppFramework\Utility\ITimeFactory;
use OCP\BackgroundJob\IJobList;
use OCP\BackgroundJob\QueuedJob;
use OCP\DB\Exception;
use Psr\Log\LoggerInterface;

/**
 * Backfills pre-existing files into recognize_pending, a batch at a time.
 *
 * The file listener only sees files as they change, so this is what makes a
 * recrawl reach content that was already on disk.
 *
 * This used to fan each file out into a per-model PHP queue consumed by the
 * TensorFlow classifier jobs. The Python daemon replaced those: it reads
 * recognize_pending and decides which classifiers apply from the file's own
 * type, so one row per file is all that is needed and the per-model
 * branching (including the landmark precondition-tag lookup) is gone.
 */
final class StorageCrawlJob extends QueuedJob {
	public const BATCH_SIZE = 2000;

	public function __construct(
		ITimeFactory $timeFactory,
		private Logger $logger,
		private IJobList $jobList,
		private StorageService $storageService,
		private PendingService $pending,
	) {
		parent::__construct($timeFactory);
	}

	/**
	 * @param array{storage_id:int, root_id:int, override_root:int, last_file_id:int, models?: list<string>} $argument
	 */
	protected function run($argument): void {
		$storageId = $argument['storage_id'];
		$rootId = $argument['root_id'];
		$overrideRoot = $argument['override_root'];
		$lastFileId = $argument['last_file_id'];
		/** @var list<string> $models */
		$models = $argument['models'] ?? Constants::MODELS;

		// Remove current iteration
		$this->jobList->remove(self::class, $argument);

		$lastSeenFileId = null;
		foreach ($this->storageService->getFilesInMount($storageId, $overrideRoot, $models, $lastFileId, self::BATCH_SIZE) as $file) {
			try {
				$this->pending->upsert((int)$file['fileid'], $storageId, PendingService::ACTION_CREATED);
				$lastSeenFileId = (int)$file['fileid'];
			} catch (Exception $e) {
				$this->logger->error('Failed to queue file for recognition', ['exception' => $e]);
				return;
			}
		}

		if ($lastSeenFileId !== null) {
			// Schedule next iteration
			$this->jobList->add(self::class, [
				'storage_id' => $storageId,
				'root_id' => $rootId,
				'override_root' => $overrideRoot,
				'last_file_id' => $lastSeenFileId,
				'models' => $models,
			]);
		}
	}
}
