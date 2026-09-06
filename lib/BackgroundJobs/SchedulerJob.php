<?php

/*
 * Copyright (c) 2021-2022 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\BackgroundJobs;

use OCA\Recognize\Constants;
use OCA\Recognize\Service\Logger;
use OCA\Recognize\Service\StorageService;
use OCP\AppFramework\Utility\ITimeFactory;
use OCP\BackgroundJob\IJobList;
use OCP\BackgroundJob\QueuedJob;
use Psr\Log\LoggerInterface;

final class SchedulerJob extends QueuedJob {
	public const INTERVAL = 30 * 60; // 30 minutes
	public const ALLOWED_MOUNT_TYPES = [
		'OC\Files\Mount\LocalHomeMountProvider',
		'OC\Files\Mount\ObjectHomeMountProvider',
		'OCA\Files_External\Config\ConfigAdapter',
		'OCA\GroupFolders\Mount\MountProvider'
	];

	public const HOME_MOUNT_TYPES = [
		'OC\Files\Mount\LocalHomeMountProvider',
		'OC\Files\Mount\ObjectHomeMountProvider',
	];

	private LoggerInterface $logger;
	private IJobList $jobList;
	private StorageService $storageService;

	public function __construct(ITimeFactory $timeFactory, Logger $logger, IJobList $jobList, StorageService $storageService) {
		parent::__construct($timeFactory);
		$this->logger = $logger;
		$this->jobList = $jobList;
		$this->storageService = $storageService;
	}

	/**
	 * @param array $argument
	 */
	protected function run($argument): void {
		/** @var list<string> $models */
		$models = $argument['models'] ?? [
			Constants::MODEL_FACES,
			Constants::MODEL_IMAGENET,
			Constants::MODEL_LANDMARKS,
			Constants::MODEL_MOVINET,
			Constants::MODEL_MUSICNN,
		];

		foreach ($this->storageService->getMounts() as $mount) {
			$this->jobList->add(StorageCrawlJob::class, [
				'storage_id' => $mount['storage_id'],
				'root_id' => $mount['root_id' ],
				'override_root' => $mount['override_root'],
				'last_file_id' => 0,
				'models' => $models,
			]);
		}

		$this->jobList->remove(self::class);
	}
}
