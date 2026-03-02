<?php

/*
 * Copyright (c) 2020-2026 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\Migration;

use Closure;
use OCA\Recognize\BackgroundJobs\SchedulerJob;
use OCA\Recognize\Classifiers\Images\ClusteringFaceClassifier;
use OCP\BackgroundJob\IJobList;
use OCP\DB\Exception;
use OCP\IDBConnection;
use OCP\Migration\IOutput;
use OCP\Migration\SimpleMigrationStep;
use Psr\Log\LoggerInterface;

/**
 * Face vector migration: clear all 128-dim face detections and clusters.
 *
 * The Python port uses InsightFace which produces 512-dim embeddings,
 * incompatible with the old 128-dim vectors from vladmandic/face-api.
 * All face data must be cleared and re-scanned.
 */
final class Version011000002Date20260301094821 extends SimpleMigrationStep {

	public function __construct(
		private IDBConnection $db,
		private IJobList $jobList,
		private LoggerInterface $logger,
	) {
	}

	public function postSchemaChange(IOutput $output, Closure $schemaClosure, array $options): void {
		try {
			$qb = $this->db->getQueryBuilder();

			// Clear all face clusters
			$output->info('Clearing face clusters for vector dimension migration...');
			$qb->delete('recognize_face_clusters')->executeStatement();

			// Clear all face detections (128-dim vectors are incompatible with 512-dim)
			$qb = $this->db->getQueryBuilder();
			$qb->delete('recognize_face_detections')->executeStatement();

			$output->info('Face data cleared. Faces will be re-scanned with 512-dim vectors.');

			// Schedule re-crawl for face classification
			$this->jobList->add(SchedulerJob::class, ['models' => [ClusteringFaceClassifier::MODEL_NAME]]);
			$output->info('Scheduled face re-classification job.');
		} catch (Exception $e) {
			$this->logger->error('Failed to clear face data for vector migration', ['exception' => $e]);
			$output->warning('Failed to clear face data: ' . $e->getMessage());
		}
	}
}
