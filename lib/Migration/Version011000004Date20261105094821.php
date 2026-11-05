<?php

declare(strict_types=1);

/*
 * Copyright (c) 2026 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */

namespace OCA\Recognize\Migration;

use Closure;
use OCP\DB\ISchemaWrapper;
use OCP\Migration\IOutput;
use OCP\Migration\SimpleMigrationStep;

/**
 * Index recognize_face_detections on (user_id, cluster_id) for the daemon's
 * per-user unclustered and cluster-sample queries.
 */
final class Version011000004Date20261105094821 extends SimpleMigrationStep {

	public function changeSchema(IOutput $output, Closure $schemaClosure, array $options): ?ISchemaWrapper {
		/** @var ISchemaWrapper $schema */
		$schema = $schemaClosure();

		if ($schema->hasTable('recognize_face_detections')) {
			$table = $schema->getTable('recognize_face_detections');
			if (!$table->hasIndex('recognize_facedet_user_cluster')) {
				$table->addIndex(['user_id', 'cluster_id'], 'recognize_facedet_user_cluster');
			}
		}

		return $schema;
	}
}
