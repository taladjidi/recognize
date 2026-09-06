<?php

/*
 * Copyright (c) 2023 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\Command;

use OCA\Recognize\BackgroundJobs\SchedulerJob;
use OCA\Recognize\BackgroundJobs\StorageCrawlJob;
use OCP\BackgroundJob\IJobList;
use Psr\Log\LoggerInterface;
use Symfony\Component\Console\Command\Command;
use Symfony\Component\Console\Input\InputInterface;
use Symfony\Component\Console\Output\OutputInterface;

final class ClearBackgroundJobs extends Command {
	private IJobList $jobList;
	private LoggerInterface $logger;
	

	public function __construct(IJobList $jobList, LoggerInterface $logger) {
		parent::__construct();
		$this->jobList = $jobList;
		$this->logger = $logger;
	}

	/**
	 * Configure the command
	 *
	 * @return void
	 */
	protected function configure() {
		$this->setName('recognize:clear-background-jobs')
			->setDescription('Remove all scheduled background jobs');
	}

	/**
	 * Execute the command
	 *
	 * @param InputInterface  $input
	 * @param OutputInterface $output
	 *
	 * @return int
	 */
	protected function execute(InputInterface $input, OutputInterface $output): int {
		try {
			// TensorFlow-era jobs: the classes are gone, so purge by name.
			foreach (Constants::LEGACY_JOB_CLASSES as $legacy) {
				$this->jobList->remove($legacy);
			}
			$this->jobList->remove(SchedulerJob::class);
			$this->jobList->remove(StorageCrawlJob::class);
		} catch (\Exception $ex) {
			$output->writeln('<error>Failed to clear background jobs</error>');
			$output->writeln($ex->getMessage());
			$this->logger->error('Failed to clear background jobs: '.$ex->getMessage(), ['exception' => $ex]);
			return 1;
		}

		return 0;
	}
}
