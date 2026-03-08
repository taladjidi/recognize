<?php

declare(strict_types=1);

/*
 * Copyright (c) 2024-2026 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */

namespace OCA\Recognize\Migration;

use Closure;
use OCP\DB\ISchemaWrapper;
use OCP\DB\Types;
use OCP\Migration\IOutput;
use OCP\Migration\SimpleMigrationStep;

/**
 * Create the unified recognize_pending table.
 *
 * Replaces the 4 fs_action tables (fs_creation, fs_deletion, fs_move,
 * fs_access_update) and 5 per-model queue tables with a single table.
 * The Python daemon polls this table directly.
 */
final class Version011000003Date20260308094821 extends SimpleMigrationStep {

	public function changeSchema(IOutput $output, Closure $schemaClosure, array $options): ?ISchemaWrapper {
		/** @var ISchemaWrapper $schema */
		$schema = $schemaClosure();

		if (!$schema->hasTable('recognize_pending')) {
			$table = $schema->createTable('recognize_pending');

			$table->addColumn('id', Types::BIGINT, [
				'autoincrement' => true,
				'notnull' => true,
				'length' => 20,
				'unsigned' => true,
			]);
			$table->addColumn('file_id', Types::BIGINT, [
				'notnull' => true,
				'length' => 20,
				'unsigned' => true,
			]);
			$table->addColumn('storage_id', Types::INTEGER, [
				'notnull' => true,
				'unsigned' => true,
			]);
			$table->addColumn('action', Types::SMALLINT, [
				'notnull' => true,
				'default' => 0,
				'comment' => '0=created, 1=updated, 2=deleted',
			]);
			$table->addColumn('added_at', Types::INTEGER, [
				'notnull' => true,
				'unsigned' => true,
			]);

			$table->setPrimaryKey(['id']);
			$table->addUniqueIndex(['file_id'], 'recognize_pending_file_id');
			$table->addIndex(['action', 'added_at'], 'recognize_pending_action_added');
		}

		return $schema;
	}
}
