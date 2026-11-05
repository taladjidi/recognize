<?php

/*
 * Copyright (c) 2026 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\BackgroundJobs;

use OCA\Recognize\Constants;
use OCA\Recognize\Service\StorageService;
use OCP\AppFramework\Services\IAppConfig;
use OCP\AppFramework\Utility\ITimeFactory;
use OCP\BackgroundJob\IJobList;
use OCP\BackgroundJob\TimedJob;
use Psr\Log\LoggerInterface;

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
		private LoggerInterface $logger,
		private IAppConfig $appConfig,
	) {
		parent::__construct($timeFactory);
		$this->setInterval(24 * 60 * 60);
		$this->setTimeSensitivity(self::TIME_INSENSITIVE);
	}

	/**
	 * Models whose `<model>.enabled` is true; all of them when no such key exists,
	 * as the daemon does (python/db.py get_enabled_models).
	 *
	 * @return list<string>
	 */
	private function enabledModels(): array {
		$set = array_filter(Constants::MODELS, fn (string $m) => $this->appConfig->hasAppKey($m . '.enabled'));
		if ($set === []) {
			return Constants::MODELS;
		}
		return array_values(array_filter($set, fn (string $m) => in_array(strtolower($this->appConfig->getAppValueString($m . '.enabled')), ['true', '1', 'yes'], true)));
	}

	protected function run($argument): void {
		// Files of a disabled model would be queued, dropped untagged, and queued again the next day.
		$models = $this->enabledModels();
		if ($models === []) {
			return;
		}
		// Stand down only for an unfinished catch-up: a stuck full recrawl must
		// not block this forever. Crawl chains hold one row per mount, so few.
		foreach ($this->jobList->getJobsIterator(StorageCrawlJob::class, null, 0) as $job) {
			$arg = $job->getArgument();
			if (is_array($arg) && ($arg['skip_tagged'] ?? false)) {
				$this->logger->info('Recognize catch-up crawl still running, not starting another');
				return;
			}
		}
		foreach ($this->storageService->getMounts() as $mount) {
			$this->jobList->add(StorageCrawlJob::class, [
				'storage_id' => $mount['storage_id'],
				'root_id' => $mount['root_id'],
				'override_root' => $mount['override_root'],
				'last_file_id' => 0,
				'models' => $models,
				'skip_tagged' => true,
			]);
		}
	}
}
