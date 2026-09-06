<?php

/*
 * Copyright (c) 2021-2022 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\BackgroundJobs;

use OCA\Recognize\Constants;
use OCA\Recognize\Db\QueueFile;
use OCA\Recognize\Service\Logger;
use OCA\Recognize\Service\QueueService;
use OCA\Recognize\Service\StorageService;
use OCA\Recognize\Service\TagManager;
use OCP\AppFramework\Utility\ITimeFactory;
use OCP\BackgroundJob\IJobList;
use OCP\BackgroundJob\QueuedJob;
use OCP\DB\Exception;
use Psr\Log\LoggerInterface;

final class StorageCrawlJob extends QueuedJob {
	public const BATCH_SIZE = 2000;
	private LoggerInterface $logger;
	private QueueService $queue;
	private IJobList $jobList;
	private TagManager $tagManager;
	private StorageService $storageService;

	public function __construct(ITimeFactory $timeFactory, Logger $logger, QueueService $queue, IJobList $jobList, TagManager $tagManager, StorageService $storageService) {
		parent::__construct($timeFactory);
		$this->logger = $logger;
		$this->queue = $queue;
		$this->jobList = $jobList;
		$this->tagManager = $tagManager;
		$this->storageService = $storageService;
	}

	/**
	 * @param array{storage_id:int, root_id:int, override_root:int, last_file_id:int, models?: list<string>} $argument
	 */
	protected function run($argument): void {
		$storageId = $argument['storage_id'];
		$rootId = $argument['root_id'];
		$overrideRoot = $argument['override_root'];
		$lastFileId = $argument['last_file_id'];
		/**
		 * @var list<string> $models
		 */
		$models = $argument['models'] ?? [
			Constants::MODEL_FACES,
			Constants::MODEL_IMAGENET,
			Constants::MODEL_LANDMARKS,
			Constants::MODEL_MOVINET,
			Constants::MODEL_MUSICNN,
		];

		// Remove current iteration
		$this->jobList->remove(self::class, $argument);

		/** @var ?QueueFile $queueFile */
		$queueFile = null;
		$i = 0;
		foreach ($this->storageService->getFilesInMount($storageId, $overrideRoot, $models, $lastFileId, self::BATCH_SIZE) as $file) {
			$i++;
			$queueFile = new QueueFile();
			$queueFile->setStorageId($storageId);
			$queueFile->setRootId($rootId);
			$queueFile->setFileId($file['fileid']);
			$queueFile->setUpdate(false);
			try {
				if ($file['image']) {
					if (in_array(Constants::MODEL_IMAGENET, $models)) {
						$this->queue->insertIntoQueue(Constants::MODEL_IMAGENET, $queueFile);
					}
					if (!in_array(Constants::MODEL_IMAGENET, $models) && in_array(Constants::MODEL_LANDMARKS, $models)) {
						$tags = $this->tagManager->getTagsForFiles([$queueFile->getFileId()]);
						$fileTags = $tags[$queueFile->getFileId()];
						$landmarkTags = array_filter($fileTags, function ($tag) {
							return in_array($tag->getName(), Constants::LANDMARKS_PRECONDITION_TAGS);
						});
						if (count($landmarkTags) > 0) {
							$this->queue->insertIntoQueue(Constants::MODEL_LANDMARKS, $queueFile);
						}
					}
					if (in_array(Constants::MODEL_FACES, $models)) {
						$this->queue->insertIntoQueue(Constants::MODEL_FACES, $queueFile);
					}
				}
				if ($file['video']) {
					$this->queue->insertIntoQueue(Constants::MODEL_MOVINET, $queueFile);
				}
				if ($file['audio']) {
					$this->queue->insertIntoQueue(Constants::MODEL_MUSICNN, $queueFile);
				}
			} catch (Exception $e) {
				$this->logger->error('Failed to add file to queue', ['exception' => $e]);
				return;
			}
		}

		if ($queueFile) {
			// Schedule next iteration
			$this->jobList->add(self::class, [
				'storage_id' => $storageId,
				'root_id' => $rootId,
				'override_root' => $overrideRoot,
				'last_file_id' => $queueFile->getFileId(),
				'models' => $models,
			]);
		}
	}
}
