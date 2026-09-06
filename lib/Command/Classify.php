<?php

/*
 * Copyright (c) 2022 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\Command;

use OCA\Recognize\Constants;
use OCA\Recognize\Db\QueueFile;
use OCA\Recognize\Exception\Exception;
use OCA\Recognize\Service\Logger;
use OCA\Recognize\Service\SettingsService;
use OCA\Recognize\Service\StorageService;
use OCA\Recognize\Service\TagManager;
use OCP\Files\Config\ICachedMountInfo;
use OCP\Files\Config\IUserMountCache;
use Symfony\Component\Console\Command\Command;
use Symfony\Component\Console\Exception\ExceptionInterface;
use Symfony\Component\Console\Input\ArrayInput;
use Symfony\Component\Console\Input\InputInterface;
use Symfony\Component\Console\Input\InputOption;
use Symfony\Component\Console\Output\OutputInterface;

final class Classify extends Command {
	/** @var array<string, Classifier>  */
	private array $classifiers = [];

	public function __construct(
		private StorageService $storageService,
		private TagManager $tagManager,
		private Logger $logger,
		ImagenetClassifier $imagenet,
		ClusteringFaceClassifier $faces,
		MovinetClassifier $movinet,
		MusicnnClassifier $musicnn,
		private IUserMountCache $userMountCache,
		private SettingsService $settings,
		private ClearBackgroundJobs $clearBackgroundJobs,
	) {
		parent::__construct();
		$this->classifiers[Constants::MODEL_IMAGENET] = $imagenet;
		$this->classifiers[Constants::MODEL_FACES] = $faces;
		$this->classifiers[Constants::MODEL_MUSICNN] = $musicnn;
		$this->classifiers[Constants::MODEL_MOVINET] = $movinet;
		// Landmarks are currently processed out of band in a background job, because imagenet schedules it directly
	}

	/**
	 * Configure the command
	 *
	 * @return void
	 */
	protected function configure() {
		$this->setName('recognize:classify')
			->setDescription('Classify all files with the current settings in one go (will likely take a long time)')
			->addOption('retry', null, InputOption::VALUE_NONE, "Only classify untagged images");
	}

	/**
	 * Execute the command
	 *
	 * @param InputInterface $input
	 * @param OutputInterface $output
	 *
	 * @return int
	 * @throws ExceptionInterface
	 */
	protected function execute(InputInterface $input, OutputInterface $output): int {
		$this->logger->setCliOutput($output);

		// pop "retry" flag from parameters passed to clear background jobs
		$clearBackgroundJobs = new ArrayInput([]);
		$this->clearBackgroundJobs->run($clearBackgroundJobs, $output);

		$models = array_values(array_filter([
			Constants::MODEL_FACES,
			Constants::MODEL_IMAGENET,
			Constants::MODEL_MOVINET,
			Constants::MODEL_MUSICNN,
		], fn ($modelName) => $this->settings->getSetting($modelName . '.enabled') === 'true'));

		$processedTag = $this->tagManager->getProcessedTag();

		foreach ($this->storageService->getMounts() as $mount) {
			$this->logger->info('Processing storage ' . $mount['storage_id'] . ' with root ID ' . $mount['override_root']);

			// Setup Filesystem for a user that can access this mount
			$mounts = array_values(array_filter($this->userMountCache->getMountsForStorageId($mount['storage_id']), function (ICachedMountInfo $mountInfo) use ($mount) {
				return $mountInfo->getRootId() === $mount['root_id'];
			}));
			if (count($mounts) > 0) {
				\OC_Util::setupFS($mounts[0]->getUser()->getUID());
			}

			$lastFileId = 0;
			do {
				$i = 0;
				$queues = [
					Constants::MODEL_IMAGENET => [],
					Constants::MODEL_FACES => [],
					Constants::MODEL_MOVINET => [],
					Constants::MODEL_MUSICNN => [],
				];
				foreach ($this->storageService->getFilesInMount($mount['storage_id'], $mount['override_root'], $models, $lastFileId) as $file) {
					$i++;
					$lastFileId = $file['fileid'];
					$queueFile = new QueueFile();
					$queueFile->setStorageId($mount['storage_id']);
					$queueFile->setRootId($mount['root_id']);
					$queueFile->setFileId($file['fileid']);
					$queueFile->setUpdate(false);

					if ($file['image']) {
						if (in_array(Constants::MODEL_FACES, $models)) {
							$queues[Constants::MODEL_FACES][] = $queueFile;
						}
					}
					// if retry flag is set, skip other classifiers for tagged files
					if ($input->getOption('retry')) {
						$fileTags = $this->tagManager->getTagsForFiles([$lastFileId]);
						// check if processed tag is already in the tags
						if (in_array($processedTag, $fileTags[$lastFileId])) {
							continue;
						}
					}
					if ($file['image']) {
						if (in_array(Constants::MODEL_IMAGENET, $models)) {
							$queues[Constants::MODEL_IMAGENET][] = $queueFile;
						}
					}
					if ($file['video']) {
						if (in_array(Constants::MODEL_MOVINET, $models)) {
							$queues[Constants::MODEL_MOVINET][] = $queueFile;
						}
					}
					if ($file['audio']) {
						if (in_array(Constants::MODEL_MUSICNN, $models)) {
							$queues[Constants::MODEL_MUSICNN][] = $queueFile;
						}
					}
				}

				foreach ($this->classifiers as $modelName => $classifier) {
					if (!in_array($modelName, $models)) {
						continue;
					}
					try {
						$classifier->setMaxExecutionTime(0);
						$classifier->classify($queues[$modelName]);
					} catch (Exception $e) {
						$this->logger->warning($e->getMessage(), ['exception' => $e]);
					} catch (\RuntimeException $e) {
						$this->logger->info('Temporary error while running ' . $modelName . 'classifier', ['exception' => $e]);
					} catch (\ErrorException $e) {
						$this->logger->info('Error while running ' . $modelName . 'classifier', ['exception' => $e]);
					}
				}
			} while ($i > 0);
			\OC_Util::tearDownFS();
		}

		return 0;
	}
}
