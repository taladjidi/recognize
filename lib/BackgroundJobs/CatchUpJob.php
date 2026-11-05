<?php

/*
 * Copyright (c) 2026 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\BackgroundJobs;

use OCA\Recognize\Service\StorageService;
use OCP\AppFramework\Utility\ITimeFactory;
use OCP\BackgroundJob\IJobList;
use OCP\BackgroundJob\TimedJob;

/**
 * Daily safety net: queue every media file the daemon has not tagged yet.
 *
 * FileListener only sees uploads and moves. Files that appear any other way
 * (occ files:scan, external storage, a sync tool) fire no event, and when the
 * listener itself broke under Nextcloud 34 nothing was queued for weeks with
 * no visible error. This crawl skips tagged files, so on a healthy instance
 * it queues nothing and costs one indexed query per batch.
 */
final class CatchUpJob extends TimedJob {
	public function __construct(
		ITimeFactory $timeFactory,
		private IJobList $jobList,
		private StorageService $storageService,
	) {
		parent::__construct($timeFactory);
		$this->setInterval(24 * 60 * 60);
		$this->setTimeSensitivity(self::TIME_INSENSITIVE);
	}

	protected function run($argument): void {
		// A crawl still running (catch-up or a full recrawl) already covers this.
		foreach ($this->jobList->getJobsIterator(StorageCrawlJob::class, 1, 0) as $_) {
			return;
		}
		foreach ($this->storageService->getMounts() as $mount) {
			$this->jobList->add(StorageCrawlJob::class, [
				'storage_id' => $mount['storage_id'],
				'root_id' => $mount['root_id'],
				'override_root' => $mount['override_root'],
				'last_file_id' => 0,
				'skip_tagged' => true,
			]);
		}
	}
}
