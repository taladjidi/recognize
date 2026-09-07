<?php

declare(strict_types=1);

/*
 * Copyright (c) 2022 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */

namespace OCA\Recognize\Controller;

use OCA\Recognize\BackgroundJobs\ClusterFacesJob;
use OCA\Recognize\BackgroundJobs\SchedulerJob;
use OCA\Recognize\BackgroundJobs\StorageCrawlJob;
use OCA\Recognize\Constants;
use OCA\Recognize\Db\FaceClusterMapper;
use OCA\Recognize\Db\FaceDetectionMapper;
use OCA\Recognize\Service\PendingService;
use OCA\Recognize\Service\SettingsService;
use OCA\Recognize\Service\TagManager;
use OCP\AppFramework\Controller;
use OCP\AppFramework\Http;
use OCP\AppFramework\Http\JSONResponse;
use OCP\AppFramework\QueryException;
use OCP\AppFramework\Services\IAppConfig;
use OCP\BackgroundJob\IJobList;
use OCP\DB\Exception;
use OCP\Exceptions\AppConfigTypeConflictException;
use OCP\IBinaryFinder;
use OCP\IRequest;

final class AdminController extends Controller {
	private TagManager $tagManager;
	private IJobList $jobList;
	private SettingsService $settingsService;
	private PendingService $pending;
	private FaceClusterMapper $clusterMapper;
	private FaceDetectionMapper $detectionMapper;
	private IAppConfig $config;
	private FaceDetectionMapper $faceDetections;
	private IBinaryFinder $binaryFinder;

	public function __construct(string $appName, IRequest $request, TagManager $tagManager, IJobList $jobList, SettingsService $settingsService, PendingService $pending, FaceClusterMapper $clusterMapper, FaceDetectionMapper $detectionMapper, IAppConfig $config, FaceDetectionMapper $faceDetections, IBinaryFinder $binaryFinder) {
		parent::__construct($appName, $request);
		$this->tagManager = $tagManager;
		$this->jobList = $jobList;
		$this->settingsService = $settingsService;
		$this->pending = $pending;
		$this->clusterMapper = $clusterMapper;
		$this->detectionMapper = $detectionMapper;
		$this->config = $config;
		$this->faceDetections = $faceDetections;
		$this->binaryFinder = $binaryFinder;
	}

	public function reset(): JSONResponse {
		$this->tagManager->resetClassifications();
		return new JSONResponse([]);
	}

	public function clearAllJobs(): JSONResponse {
		// Purge rows left over from the TensorFlow-era jobs. Those classes no
		// longer exist, so they are removed by name rather than by ::class.
		foreach (Constants::LEGACY_JOB_CLASSES as $legacy) {
			$this->jobList->remove($legacy);
		}
		$this->jobList->remove(ClusterFacesJob::class);
		$this->jobList->remove(SchedulerJob::class);
		$this->jobList->remove(StorageCrawlJob::class);
		return new JSONResponse([]);
	}

	public function recrawl(): JSONResponse {
		$this->clearAllJobs();
		$this->jobList->add(SchedulerJob::class);
		return new JSONResponse([]);
	}

	public function resetFaces(): JSONResponse {
		try {
			$this->clusterMapper->deleteAll();
			$this->detectionMapper->deleteAll();
		} catch (Exception $e) {
			return new JSONResponse([], Http::STATUS_INTERNAL_SERVER_ERROR);
		}
		return new JSONResponse([]);
	}

	/**
	 * Health of the Python inference daemon.
	 *
	 * This replaces the TensorFlow/Node probes (libtensorflow, wasmtensorflow,
	 * gputensorflow, nodejs, avx, musl, platform). That runtime was removed
	 * when classification moved to the daemon, so those checks could only ever
	 * report failure — and the settings UI was gated behind them.
	 */
	public function daemon(): JSONResponse {
		$lastSeen = (int)$this->config->getAppValueString('daemon.lastSeen', '0');
		$pending = $this->pending->countByMediaType();
		return new JSONResponse([
			'lastSeen' => $lastSeen ?: null,
			'ageSeconds' => $lastSeen ? (time() - $lastSeen) : null,
			// The daemon heartbeats every poll; a few minutes of silence means
			// it is stopped or wedged.
			'healthy' => $lastSeen > 0 && (time() - $lastSeen) < 600,
			'pending' => array_sum($pending),
		]);
	}

	public function count(): JSONResponse {
		$count = count($this->tagManager->findClassifiedFiles());
		return new JSONResponse(['count' => $count]);
	}

	public function countMissed(): JSONResponse {
		$count = count($this->tagManager->findMissedClassifications());
		return new JSONResponse(['count' => $count]);
	}

	public function countQueued(): JSONResponse {
		// The per-model PHP queues are gone: the Python daemon reads
		// recognize_pending, which holds one row per file, and picks
		// classifiers from the file's own type. Report the same split here.
		$pending = $this->pending->countByMediaType();
		return new JSONResponse([
			'imagenet' => $pending['image'],
			'faces' => $pending['image'],
			'landmarks' => $pending['image'],
			'movinet' => $pending['video'],
			'musicnn' => $pending['audio'],
			'clusterFaces' => $this->faceDetections->countUnclustered(),
		]);
	}

	public function hasJobs(string $task): JSONResponse {
		// Classification is no longer done by per-model background jobs. What
		// is schedulable now is the crawl that feeds recognize_pending, so
		// every model task maps to it.
		$tasks = [
			'faces' => StorageCrawlJob::class,
			'imagenet' => StorageCrawlJob::class,
			'landmarks' => StorageCrawlJob::class,
			'musicnn' => StorageCrawlJob::class,
			'movinet' => StorageCrawlJob::class,
			'clusterFaces' => ClusterFacesJob::class,
		];
		if (!isset($tasks[$task])) {
			return new JSONResponse([], Http::STATUS_BAD_REQUEST);
		}
		$iterator = $this->jobList->getJobsIterator($tasks[$task], null, 0);
		$lastRun = [];
		foreach ($iterator as $job) {
			$lastRun[] = $job->getLastRun();
		}
		$count = count($lastRun);
		$lastRun = $lastRun? max($lastRun) : 0;
		return new JSONResponse([
			'scheduled' => $count,
			'lastRun' => $lastRun,
		]);
	}






	public function ffmpeg(): JSONResponse {
		try {
			exec($this->settingsService->getSetting('ffmpeg_binary') . ' -version' . ' 2>&1', $output, $returnCode);
		} catch (\Throwable $e) {
			return new JSONResponse(['ffmpeg' => null]);
		}

		if ($returnCode !== 0) {
			return new JSONResponse(['ffmpeg' => false]);
		}

		if (preg_match('/version (.*?) /', trim($output[0]), $matches) !== 1) {
			return new JSONResponse(['ffmpeg' => false]);
		}
		$version = $matches[1];
		return new JSONResponse(['ffmpeg' => $version]);
	}

	public function python(): JSONResponse {
		$pythonBinary = $this->settingsService->getSetting('python_binary');
		if ($pythonBinary === '') {
			return new JSONResponse(['python' => null]);
		}
		try {
			exec($pythonBinary . ' --version' . ' 2>&1', $output, $returnCode);
		} catch (\Throwable $e) {
			return new JSONResponse(['python' => null]);
		}

		if ($returnCode !== 0) {
			return new JSONResponse(['python' => false]);
		}

		$version = trim(implode("\n", $output));
		return new JSONResponse(['python' => $version]);
	}





	public function cron(): JSONResponse {
		try {
			/** @var IAppConfig $appConfig */
			$appConfig = \OC::$server->getRegisteredAppContainer('core')->get(IAppConfig::class);
			$cron = $appConfig->getAppValueString('backgroundjobs_mode', '');
		} catch (QueryException|AppConfigTypeConflictException $e) {
			return new JSONResponse([], Http::STATUS_INTERNAL_SERVER_ERROR);
		}
		return new JSONResponse(['cron' => $cron]);
	}

	/**
	 * @param string $setting
	 * @param scalar $value
	 * @return JSONResponse
	 */
	public function setSetting(string $setting, float|bool|int|string $value): JSONResponse {
		try {
			$this->settingsService->setSetting($setting, (string) $value);
			return new JSONResponse([], Http::STATUS_OK);
		} catch (\OCA\Recognize\Exception\Exception $e) {
			return new JSONResponse([], Http::STATUS_BAD_REQUEST);
		}
	}

	public function getSetting(string $setting): JSONResponse {
		return new JSONResponse(['value' => $this->settingsService->getSetting($setting)]);
	}
}
