<?php

/*
 * Copyright (c) 2022 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\Db;

use OCA\Recognize\Service\FaceClusterAnalyzer;
use OCP\AppFramework\Db\DoesNotExistException;
use OCP\AppFramework\Db\Entity;
use OCP\AppFramework\Db\QBMapper;
use OCP\DB\Exception;
use OCP\DB\QueryBuilder\IQueryBuilder;
use OCP\IConfig;
use OCP\IDBConnection;

/**
 * @template-extends QBMapper<FaceDetection>
 */
final class FaceDetectionMapper extends QBMapper {
	private IConfig $config;

	public function __construct(IDBConnection $db, IConfig $config) {
		parent::__construct($db, 'recognize_face_detections', FaceDetection::class);
		$this->db = $db;
		$this->config = $config;
	}

	/**
	 * @throws \OCP\AppFramework\Db\DoesNotExistException
	 * @throws \OCP\AppFramework\Db\MultipleObjectsReturnedException
	 * @throws \OCP\DB\Exception
	 */
	public function find(int $id): FaceDetection {
		$qb = $this->db->getQueryBuilder();
		$qb
			->select(FaceDetection::$columns)
			->from('recognize_face_detections')
			->where($qb->expr()->eq('id', $qb->createNamedParameter($id)));

		return $this->findEntity($qb);
	}

	/**
	 * @param FaceDetection $entity
	 * @throws \OCP\DB\Exception
	 */
	public function insert(Entity $entity): FaceDetection {
		/** @var FaceDetection $entity */
		$qb = $this->db->getQueryBuilder();
		$qb->select(FaceDetection::$columns)
			->from('recognize_face_detections')
			->where($qb->expr()->eq('file_id', $qb->createPositionalParameter($entity->getFileId(), IQueryBuilder::PARAM_INT)))
			->andWhere($qb->expr()->eq('user_id', $qb->createPositionalParameter($entity->getUserId(), IQueryBuilder::PARAM_STR)))
			->andWhere($qb->expr()->eq('x', $qb->createPositionalParameter($entity->getX(), IQueryBuilder::PARAM_INT)))
			->andWhere($qb->expr()->eq('y', $qb->createPositionalParameter($entity->getY(), IQueryBuilder::PARAM_INT)))
			->andWhere($qb->expr()->eq('height', $qb->createPositionalParameter($entity->getHeight(), IQueryBuilder::PARAM_INT)))
			->andWhere($qb->expr()->eq('width', $qb->createPositionalParameter($entity->getWidth(), IQueryBuilder::PARAM_INT)));
		$duplicates = $this->findEntities($qb);

		if (empty($duplicates)) {
			return parent::insert($entity);
		}

		return $duplicates[0];
	}

	/**
	 * @throws Exception
	 */
	public function insertWithoutDeduplication(Entity $entity): FaceDetection {
		/** @var FaceDetection $entity */
		return parent::insert($entity);
	}

	/**
	 * @throws \OCP\DB\Exception
	 */
	public function deleteAll(): void {
		$qb = $this->db->getQueryBuilder();
		$qb->delete('recognize_face_detections')
			->executeStatement();
	}

	/**
	 * @param int $clusterId
	 * @return list<\OCA\Recognize\Db\FaceDetection>
	 * @throws \OCP\DB\Exception
	 */
	public function findByClusterId(int $clusterId) : array {
		$qb = $this->db->getQueryBuilder();
		$qb->select(FaceDetection::$columns)
			->from('recognize_face_detections')
			->where($qb->expr()->eq('cluster_id', $qb->createPositionalParameter($clusterId)));
		return $this->findEntities($qb);
	}

	/**
	 * Like findByClusterId but filters out detections whose files no longer exist
	 * and restricts to a specific user's detections.
	 * Uses a single INNER JOIN instead of N individual file lookups.
	 */
	private const LIGHT_COLUMNS = ['id', 'user_id', 'file_id', 'x', 'y', 'height', 'width', 'cluster_id', 'threshold'];

	public function findByClusterIdWithExistingFiles(int $clusterId, string $userId) : array {
		$qb = $this->db->getQueryBuilder();
		$columns = array_map(fn ($col) => 'd.'.$col, self::LIGHT_COLUMNS);
		$qb->select($columns)
			->from('recognize_face_detections', 'd')
			->innerJoin('d', 'filecache', 'fc', $qb->expr()->eq('d.file_id', 'fc.fileid'))
			->where($qb->expr()->eq('d.cluster_id', $qb->createPositionalParameter($clusterId)))
			->andWhere($qb->expr()->eq('d.user_id', $qb->createPositionalParameter($userId)))
			->andWhere($qb->expr()->like('fc.path', $qb->createPositionalParameter('files/%')));
		return $this->findEntities($qb);
	}

	/**
	 * Batch count detections per cluster for a user. Single GROUP BY query
	 * replaces N individual countByClusterId calls.
	 * @return array<int, int> cluster_id => count
	 */
	public function batchCountByUser(string $userId): array {
		$qb = $this->db->getQueryBuilder();
		$qb->select('d.cluster_id', $qb->func()->count('d.id', 'cnt'))
			->from('recognize_face_detections', 'd')
			->innerJoin('d', 'filecache', 'fc', $qb->expr()->eq('d.file_id', 'fc.fileid'))
			->where($qb->expr()->eq('d.user_id', $qb->createPositionalParameter($userId)))
			->andWhere($qb->expr()->isNotNull('d.cluster_id'))
			->andWhere($qb->expr()->gt('d.cluster_id', $qb->createPositionalParameter(0)))
			->andWhere($qb->expr()->like('fc.path', $qb->createPositionalParameter('files/%')))
			->groupBy('d.cluster_id');
		$result = $qb->executeQuery();
		$counts = [];
		while ($row = $result->fetch()) {
			$counts[(int)$row['cluster_id']] = (int)$row['cnt'];
		}
		$result->closeCursor();
		return $counts;
	}

	/**
	 * Batch fetch one preview detection per cluster for a user. Single query
	 * replaces N individual findDetectionForPreviewImageByClusterId calls.
	 * Picks the largest face (by height*width) with an existing non-trashed file.
	 * @return array<int, FaceDetection> cluster_id => detection
	 */
	public function batchPreviewsByUser(string $userId): array {
		// Use a subquery to find the best detection per cluster
		$qb = $this->db->getQueryBuilder();
		$columns = array_map(fn ($col) => 'd.'.$col, self::LIGHT_COLUMNS);
		$qb->select($columns)
			->from('recognize_face_detections', 'd')
			->innerJoin('d', 'filecache', 'fc', $qb->expr()->eq('d.file_id', 'fc.fileid'))
			->where($qb->expr()->eq('d.user_id', $qb->createPositionalParameter($userId)))
			->andWhere($qb->expr()->isNotNull('d.cluster_id'))
			->andWhere($qb->expr()->gt('d.cluster_id', $qb->createPositionalParameter(0)))
			->andWhere($qb->expr()->like('fc.path', $qb->createPositionalParameter('files/%')))
			->addOrderBy('d.cluster_id')
			->addOrderBy('d.height', 'DESC')
			->addOrderBy('d.width', 'DESC');
		$result = $qb->executeQuery();

		$previews = [];
		while ($row = $result->fetch()) {
			$clusterId = (int)$row['cluster_id'];
			if (!isset($previews[$clusterId])) {
				try {
					$previews[$clusterId] = $this->mapRowToEntity($row);
				} catch (\Exception $e) {
					// skip unmappable rows
				}
			}
		}
		$result->closeCursor();
		return $previews;
	}

	/**
	 * Find a detection in a cluster by the real filename (from filecache).
	 * Used when the Photos app sends MOVE requests with real filenames
	 * instead of detection-prefixed basenames.
	 */
	public function findByClusterIdAndFileName(int $clusterId, string $userId, string $fileName) : FaceDetection {
		$qb = $this->db->getQueryBuilder();
		$columns = array_map(fn ($col) => 'd.'.$col, FaceDetection::$columns);
		$qb->select($columns)
			->from('recognize_face_detections', 'd')
			->innerJoin('d', 'filecache', 'fc', $qb->expr()->eq('d.file_id', 'fc.fileid'))
			->where($qb->expr()->eq('d.cluster_id', $qb->createPositionalParameter($clusterId)))
			->andWhere($qb->expr()->eq('d.user_id', $qb->createPositionalParameter($userId)))
			->andWhere($qb->expr()->eq('fc.name', $qb->createPositionalParameter($fileName)))
			->setMaxResults(1);
		return $this->findEntity($qb);
	}

	/**
	 * Find a rejected detection (cluster_id = -1) by user and real filename.
	 */
	public function findRejectedByUserIdAndFileName(string $userId, string $fileName) : FaceDetection {
		$qb = $this->db->getQueryBuilder();
		$columns = array_map(fn ($col) => 'd.'.$col, FaceDetection::$columns);
		$qb->select($columns)
			->from('recognize_face_detections', 'd')
			->innerJoin('d', 'filecache', 'fc', $qb->expr()->eq('d.file_id', 'fc.fileid'))
			->where($qb->expr()->eq('d.cluster_id', $qb->createPositionalParameter(-1)))
			->andWhere($qb->expr()->eq('d.user_id', $qb->createPositionalParameter($userId)))
			->andWhere($qb->expr()->eq('fc.name', $qb->createPositionalParameter($fileName)))
			->setMaxResults(1);
		return $this->findEntity($qb);
	}

	/** Rejected (unassigned) detections whose file still sits in the user's files. */
	public function countRejectedByUserId(string $userId) : int {
		$qb = $this->db->getQueryBuilder();
		$qb->select($qb->func()->count('d.id'))
			->from('recognize_face_detections', 'd')
			->innerJoin('d', 'filecache', 'fc', $qb->expr()->eq('d.file_id', 'fc.fileid'))
			->where($qb->expr()->eq('d.cluster_id', $qb->createPositionalParameter(-1)))
			->andWhere($qb->expr()->eq('d.user_id', $qb->createPositionalParameter($userId)))
			->andWhere($qb->expr()->like('fc.path', $qb->createPositionalParameter('files/%')));
		$result = $qb->executeQuery();
		$count = $result->fetch(\PDO::FETCH_COLUMN);
		$result->closeCursor();
		return (int) $count;
	}

	public function countByClusterId(int $clusterId, string $userId) : int {
		$qb = $this->db->getQueryBuilder();
		$qb->select($qb->func()->count('d.id'))
			->from('recognize_face_detections', 'd')
			->innerJoin('d', 'filecache', 'fc', $qb->expr()->eq('d.file_id', 'fc.fileid'))
			->where($qb->expr()->eq('d.cluster_id', $qb->createPositionalParameter($clusterId)))
			->andWhere($qb->expr()->eq('d.user_id', $qb->createPositionalParameter($userId)))
			->andWhere($qb->expr()->like('fc.path', $qb->createPositionalParameter('files/%')));
		$result = $qb->executeQuery();
		$count = $result->fetch(\PDO::FETCH_COLUMN);
		$result->closeCursor();
		return (int) $count;
	}

	/**
	 * @throws \OCP\DB\Exception
	 * @return list<\OCA\Recognize\Db\FaceDetection>
	 */
	public function findByUserId(string $userId): array {
		$qb = $this->db->getQueryBuilder();
		$qb->select(FaceDetection::$columns)
			->from('recognize_face_detections')
			->where($qb->expr()->eq('user_id', $qb->createPositionalParameter($userId)));
		return $this->findEntities($qb);
	}

	/**
	 * @throws \OCP\DB\Exception
	 * @return list<\OCA\Recognize\Db\FaceDetection>
	 */
	public function findByFileId(int $fileId): array {
		$qb = $this->db->getQueryBuilder();
		$qb->select(FaceDetection::$columns)
			->from('recognize_face_detections')
			->where($qb->expr()->eq('file_id', $qb->createPositionalParameter($fileId, IQueryBuilder::PARAM_INT)));
		return $this->findEntities($qb);
	}

	/**
	 * @throws \OCP\DB\Exception
	 * @return \OCA\Recognize\Db\FaceDetection[]
	 */
	public function findByFileIdAndUser(int $fileId, string $userId): array {
		$qb = $this->db->getQueryBuilder();
		$qb->select(FaceDetection::$columns)
			->from('recognize_face_detections')
			->where($qb->expr()->eq('file_id', $qb->createPositionalParameter($fileId, IQueryBuilder::PARAM_INT)))
			->andWhere($qb->expr()->eq('user_id', $qb->createPositionalParameter($userId, IQueryBuilder::PARAM_STR)));
		return $this->findEntities($qb);
	}

	/**
	 * @throws \OCP\DB\Exception
	 */
	public function copyDetectionsForFileFromUserToUser(int $fileId, string $fromUser, string $toUser) : void {
		$detections = $this->findByFileIdAndUser($fileId, $fromUser);
		foreach ($detections as $detection) {
			$detectionCopy = new FaceDetection();
			$detectionCopy->setX($detection->getX());
			$detectionCopy->setY($detection->getY());
			$detectionCopy->setVector($detection->getVector());
			$detectionCopy->setFileId($detection->getFileId());
			$detectionCopy->setHeight($detection->getHeight());
			$detectionCopy->setWidth($detection->getWidth());
			$detectionCopy->setUserId($toUser);
			$this->insert($detectionCopy);
		}
	}

	/**
	 * @param int $fileId
	 * @param string[] $userIds
	 * @return void
	 * @throws \OCP\DB\Exception
	 */
	public function removeDetectionsForFileFromUsersNotInList(int $fileId, array $userIds) : void {
		$qb = $this->db->getQueryBuilder();
		$qb->delete('recognize_face_detections')
			->where($qb->expr()->eq('file_id', $qb->createPositionalParameter($fileId, IQueryBuilder::PARAM_INT)))
			->andWhere($qb->expr()->notIn('user_id', $qb->createPositionalParameter($userIds, IQueryBuilder::PARAM_STR_ARRAY)));
		$qb->executeStatement();
	}

	/**
	 * @param \OCA\Recognize\Db\FaceDetection $faceDetection
	 * @param \OCA\Recognize\Db\FaceCluster $faceCluster
	 * @return void
	 * @throws \OCP\DB\Exception
	 */
	public function assocWithCluster(FaceDetection $faceDetection, FaceCluster $faceCluster) {
		$faceDetection->setClusterId($faceCluster->getId());
		$this->update($faceDetection);
	}

	/**
	 * @param int $fileId
	 * @return list<\OCA\Recognize\Db\FaceDetectionWithTitle>
	 * @throws \OCP\DB\Exception
	 */
	/**
	 * Detections (with cluster title) for many files in one query.
	 *
	 * The DAV face-detections property is emitted per child, so listing a
	 * cluster of N photos previously ran N of these queries. Measured on a
	 * 1331-photo cluster: 314 ms as N+1, 25 ms batched.
	 *
	 * @param list<int> $fileIds
	 * @return array<int, list<\OCA\Recognize\Db\FaceDetectionWithTitle>> keyed by file id
	 */
	public function findByFileIdsWithTitle(array $fileIds, string $userId) : array {
		if ($fileIds === []) {
			return [];
		}

		$lightWithTitle = array_merge(
			array_map(fn ($col) => 'd.'.$col, self::LIGHT_COLUMNS),
			['c.title']
		);

		$byFile = [];
		// Chunked to stay clear of parameter limits on large clusters.
		foreach (array_chunk(array_values(array_unique($fileIds)), 900) as $chunk) {
			$qb = $this->db->getQueryBuilder();
			$qb->select($lightWithTitle)
				->from('recognize_face_detections', 'd')
				->where($qb->expr()->in('d.file_id', $qb->createNamedParameter($chunk, IQueryBuilder::PARAM_INT_ARRAY)))
				->andWhere($qb->expr()->eq('d.user_id', $qb->createNamedParameter($userId)))
				->leftJoin('d', 'recognize_face_clusters', 'c', $qb->expr()->eq('d.cluster_id', 'c.id'));
			/** @var list<\OCA\Recognize\Db\FaceDetectionWithTitle> $entities */
			$entities = $this->findEntities($qb);
			foreach ($entities as $entity) {
				$byFile[$entity->getFileId()][] = $entity;
			}
		}

		return $byFile;
	}

	/**
	 * @return list<\OCA\Recognize\Db\FaceDetectionWithTitle>
	 */
	public function findByFileIdWithTitle(int $fileId, string $userId) : array {
		$qb = $this->db->getQueryBuilder();
		$lightWithTitle = array_merge(
			array_map(fn ($col) => 'd.'.$col, self::LIGHT_COLUMNS),
			['c.title']
		);
		$qb->select($lightWithTitle)
			->from('recognize_face_detections', 'd')
			->where($qb->expr()->eq('d.file_id', $qb->createPositionalParameter($fileId, IQueryBuilder::PARAM_INT)))
			->andWhere($qb->expr()->eq('d.user_id', $qb->createPositionalParameter($userId)))
			->leftJoin('d', 'recognize_face_clusters', 'c', $qb->expr()->eq('d.cluster_id', 'c.id'));
		/** @var list<\OCA\Recognize\Db\FaceDetectionWithTitle> $entities */
		$entities = $this->findEntities($qb);
		return $entities;
	}

	/**
	 * @return list<string>
	 * @throws \OCP\DB\Exception
	 */
	public function findUserIds() :array {
		$qb = $this->db->getQueryBuilder();
		$qb->selectDistinct('d.user_id')
			->from('recognize_face_detections', 'd');
		/** @var list<string> */
		$list = $qb->executeQuery()->fetchAll(\PDO::FETCH_COLUMN);
		return $list;
	}

	/**
	 * @param string $userId
	 * @return \OCA\Recognize\Db\FaceDetection[]
	 * @throws \OCP\DB\Exception
	 */
	public function findUnclusteredByUserId(string $userId, int $limit = 0, float $minHeight = 0, float $minWidth = 0) : array {
		$qb = $this->db->getQueryBuilder();
		$qb->select(FaceDetection::$columns)
			->from('recognize_face_detections')
			->where($qb->expr()->eq('user_id', $qb->createPositionalParameter($userId)))
			->andWhere($qb->expr()->gte('height', $qb->createPositionalParameter($minHeight)))
			->andWhere($qb->expr()->gte('width', $qb->createPositionalParameter($minWidth)))
			->andWhere($qb->expr()->isNull('cluster_id'));
		if ($limit > 0) {
			$qb->setMaxResults($limit);
		}
		return $this->findEntities($qb);
	}

	/**
	 * @param string $userId
	 * @return \OCA\Recognize\Db\FaceDetection[]
	 * @throws \OCP\DB\Exception
	 */
	public function findRejectedByUserId(string $userId) : array {
		$qb = $this->db->getQueryBuilder();
		$qb->select(FaceDetection::$columns)
			->from('recognize_face_detections')
			->where($qb->expr()->eq('user_id', $qb->createPositionalParameter($userId)))
			->andWhere($qb->expr()->eq('cluster_id', $qb->createPositionalParameter(-1)));
		return $this->findEntities($qb);
	}

	/**
	 * @param int $clusterId
	 * @param int $n
	 * @param int $minHeight
	 * @param int $minWidth
	 * @return FaceDetection[]
	 * @throws \OCP\DB\Exception
	 */
	public function findClusterSample(int $clusterId, int $n, int $minHeight = 0, int $minWidth = 0): array {
		$qb = $this->db->getQueryBuilder();
		$qb->select(FaceDetection::$columns)
			->from('recognize_face_detections', 'd')
			->where($qb->expr()->eq('cluster_id', $qb->createPositionalParameter($clusterId)))
			->andWhere($qb->expr()->gte('height', $qb->createPositionalParameter($minHeight)))
			->andWhere($qb->expr()->gte('width', $qb->createPositionalParameter($minWidth)))
			->orderBy(
				$qb->createFunction(
					$this->config->getSystemValue('dbtype', 'sqlite') === 'mysql'
						? 'RAND()'
						: 'RANDOM()'
				)
			)
			->setMaxResults($n);
		return $this->findEntities($qb);
	}

	/**
	 * @param string $userId
	 * @param int $n
	 * @param float $minHeight
	 * @param float $minWidth
	 * @return FaceDetection[]
	 * @throws \OCP\DB\Exception
	 */
	public function sampleRejectedDetectionsByUserId(string $userId, int $n, float $minHeight = 0, float $minWidth = 0): array {
		$qb = $this->db->getQueryBuilder();
		$qb->select(FaceDetection::$columns)
			->from('recognize_face_detections', 'd')
			->where($qb->expr()->eq('cluster_id', $qb->createPositionalParameter(-1)))
			->andWhere($qb->expr()->eq('user_id', $qb->createPositionalParameter($userId)))
			->andWhere($qb->expr()->gte('height', $qb->createPositionalParameter($minHeight)))
			->andWhere($qb->expr()->gte('width', $qb->createPositionalParameter($minWidth)))
			->orderBy(
				$qb->createFunction(
					$this->config->getSystemValue('dbtype', 'sqlite') === 'mysql'
						? 'RAND()'
						: 'RANDOM()'
				)
			)
			->setMaxResults($n);
		return $this->findEntities($qb);
	}

	public function removeAllClusters(): void {
		$qb = $this->db->getQueryBuilder();
		$qb->update('recognize_face_detections')
			->set('cluster_id', $qb->createPositionalParameter(null))
			->where($qb->expr()->isNotNull('cluster_id'));
		$qb->executeStatement();
	}

	public function findDetectionForPreviewImageByClusterId(int $clusterId, string $userId) : FaceDetection {
		$qb = $this->db->getQueryBuilder();
		$lightColumns = ['id', 'user_id', 'file_id', 'x', 'y', 'height', 'width', 'cluster_id', 'threshold'];
		$qb->select(array_map(fn ($col) => 'd.'.$col, $lightColumns))
			->from('recognize_face_detections', 'd')
			->innerJoin('d', 'filecache', 'fc', $qb->expr()->eq('d.file_id', 'fc.fileid'))
			->where($qb->expr()->eq('d.cluster_id', $qb->createPositionalParameter($clusterId)))
			->andWhere($qb->expr()->eq('d.user_id', $qb->createPositionalParameter($userId)))
			->andWhere($qb->expr()->gt($qb->createFunction('d.x - d.width * 0.5'), $qb->createFunction('0')))
			->andWhere($qb->expr()->gt($qb->createFunction('d.y - d.height * 0.5'), $qb->createFunction('0')))
			->andWhere($qb->expr()->gt($qb->createFunction('1 - (d.x + d.width) - d.width * 0.5'), $qb->createFunction('0')))
			->andWhere($qb->expr()->gt($qb->createFunction('1 - (d.y + d.height) - d.height * 0.5'), $qb->createFunction('0')))
			->groupBy('d.id')
			->addOrderBy('d.height', 'DESC')
			->addOrderBy('d.width', 'DESC')
			->setMaxResults(1);
		try {
			return $this->findEntity($qb);
		} catch (DoesNotExistException $e) {
			$qb = $this->db->getQueryBuilder();
			$lightColumns = ['id', 'user_id', 'file_id', 'x', 'y', 'height', 'width', 'cluster_id', 'threshold'];
		$qb->select(array_map(fn ($col) => 'd.'.$col, $lightColumns))
				->from('recognize_face_detections', 'd')
				->innerJoin('d', 'filecache', 'fc', $qb->expr()->eq('d.file_id', 'fc.fileid'))
				->where($qb->expr()->eq('d.cluster_id', $qb->createPositionalParameter($clusterId)))
				->andWhere($qb->expr()->eq('d.user_id', $qb->createPositionalParameter($userId)))
				->groupBy('d.id')
				->addOrderBy('d.height', 'DESC')
				->addOrderBy('d.width', 'DESC')
				->setMaxResults(1);
			return $this->findEntity($qb);
		}
	}

	public function countUnclustered(): int {
		$qb = $this->db->getQueryBuilder();
		$qb->select($qb->func()->count('id'))
			->from('recognize_face_detections')
			->where($qb->expr()->isNull('cluster_id'))
			->andWhere($qb->expr()->gte('height', $qb->createPositionalParameter(FaceClusterAnalyzer::MIN_DETECTION_SIZE)))
			->andWhere($qb->expr()->gte('width', $qb->createPositionalParameter(FaceClusterAnalyzer::MIN_DETECTION_SIZE)));
		$result = $qb->executeQuery();
		/** @var int|string $count */
		$count = $result->fetch(\PDO::FETCH_COLUMN);
		$result->closeCursor();
		return (int) $count;
	}

	/**
	 * @return array<string>
	 * @throws \OCP\DB\Exception
	 */
	public function getUsersForUnclustered(): array {
		$qb = $this->db->getQueryBuilder();
		$qb->selectDistinct('user_id')
			->from('recognize_face_detections')
			->where($qb->expr()->isNull('cluster_id'))
			->andWhere($qb->expr()->gte('height', $qb->createPositionalParameter(FaceClusterAnalyzer::MIN_DETECTION_SIZE)))
			->andWhere($qb->expr()->gte('width', $qb->createPositionalParameter(FaceClusterAnalyzer::MIN_DETECTION_SIZE)));
		$result = $qb->executeQuery();
		/** @var array<string> $users */
		$users = $result->fetchAll(\PDO::FETCH_COLUMN);
		$result->closeCursor();
		return $users;
	}

	protected function mapRowToEntity(array $row): Entity {
		try {
			return parent::mapRowToEntity($row);
		} catch (\Exception $e) {
			$entity = FaceDetectionWithTitle::fromRow($row);
			if ($entity->getTitle() === '') {
				$entity->setTitle((string) $entity->getClusterId());
			}
			return $entity;
		}
	}
}
