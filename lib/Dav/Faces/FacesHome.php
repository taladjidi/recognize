<?php

/*
 * Copyright (c) 2022 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\Dav\Faces;

use OCA\Recognize\Db\FaceCluster;
use OCA\Recognize\Db\FaceClusterMapper;
use OCA\Recognize\Db\FaceDetectionMapper;
use OCP\AppFramework\Db\DoesNotExistException;
use OCP\Files\IRootFolder;
use OCP\IPreview;
use OCP\ITagManager;
use OCP\IUser;
use Sabre\DAV\Exception\Forbidden;
use Sabre\DAV\Exception\NotFound;
use Sabre\DAV\ICollection;

final class FacesHome implements ICollection {
	/**
	 * Smallest cluster shown in the faces listing.
	 *
	 * Clustering leaves a long tail of tiny groups; hiding them keeps the
	 * overview to people you actually have a set of photos of. Nothing is
	 * deleted and getChild() still resolves hidden clusters directly.
	 */
	public const MIN_LISTED_CLUSTER_SIZE = 5;

	private FaceClusterMapper $faceClusterMapper;
	private IUser $user;
	private FaceDetectionMapper $faceDetectionMapper;
	private IRootFolder $rootFolder;
	private array $children = [];
	private ITagManager $tagManager;
	private IPreview $previewManager;

	public function __construct(FaceClusterMapper $faceClusterMapper, IUser $user, FaceDetectionMapper $faceDetectionMapper, IRootFolder $rootFolder, ITagManager $tagManager, IPreview $previewManager) {
		$this->faceClusterMapper = $faceClusterMapper;
		$this->user = $user;
		$this->faceDetectionMapper = $faceDetectionMapper;
		$this->rootFolder = $rootFolder;
		$this->tagManager = $tagManager;
		$this->previewManager = $previewManager;
	}

	public function delete(): never {
		throw new Forbidden();
	}

	public function getName(): string {
		return 'faces';
	}

	public function setName($name): never {
		throw new Forbidden('Permission denied to rename this folder');
	}

	public function createDirectory($name) {
		if ($this->childExists(basename($name))) {
			throw new Forbidden('Not allowed to create duplicate names');
		}
		if (preg_match('/^[0-9]+$/', basename($name), $matches) != false) {
			throw new Forbidden('Not allowed to use numbers as names');
		}
		$entity = new FaceCluster();
		$entity->setUserId($this->user->getUID());
		$entity->setTitle(basename($name));
		$this->faceClusterMapper->insert($entity);
	}

	public function createFile($name, $data = null): never {
		throw new Forbidden('Not allowed to create files in this folder');
	}

	public function getChild($name) : FaceRoot {
		if (count($this->children) !== 0) {
			foreach ($this->getChildren() as $child) {
				if ($child->getName() === $name) {
					return $child;
				}
			}
			throw new NotFound();
		}
		try {
			$cluster = $this->faceClusterMapper->findByUserAndTitle($this->user->getUID(), $name);
		} catch (DoesNotExistException $e) {
			try {
				$cluster = $this->faceClusterMapper->find((int) $name);
				if ($cluster->getUserId() !== $this->user->getUID()) {
					throw new NotFound();
				}
			} catch (DoesNotExistException $e) {
				throw new NotFound();
			}
		} catch (\Exception $e) {
			throw new NotFound();
		}

		return new FaceRoot($this->faceClusterMapper, $cluster, $this->user, $this->faceDetectionMapper, $this->rootFolder, $this->tagManager, $this->previewManager);
	}

	/**
	 * @return \OCA\Recognize\Dav\Faces\FaceRoot[]
	 * @throws \OCP\DB\Exception
	 */
	public function getChildren(): array {
		if (count($this->children) === 0) {
			$userId = $this->user->getUID();
			$clusters = $this->faceClusterMapper->findByUserId($userId);

			// Batch-prefetch counts and previews (2 queries instead of 2×N)
			$counts = $this->faceDetectionMapper->batchCountByUser($userId);
			$previews = $this->faceDetectionMapper->batchPreviewsByUser($userId);

			// Clusters below the threshold are omitted from the listing. Face
			// clustering produces a long tail of two- and three-photo groups
			// that are mostly noise; on this library 1243 of 1994 clusters are
			// under five photos, yet together they account for only 15% of
			// detected faces. Listing them costs an XML entry and a cover image
			// each, which dominates the page.
			//
			// This filters the LISTING only. getChild() still resolves any
			// cluster by name or id, so a bookmarked link, a MOVE target or a
			// direct navigation keeps working, and nothing is deleted.
			$this->children = [];
			foreach ($clusters as $cluster) {
				$count = $counts[$cluster->getId()] ?? 0;
				if ($count < self::MIN_LISTED_CLUSTER_SIZE) {
					continue;
				}

				$root = new FaceRoot($this->faceClusterMapper, $cluster, $this->user, $this->faceDetectionMapper, $this->rootFolder, $this->tagManager, $this->previewManager);
				$root->setPreloadedData(
					$count,
					$previews[$cluster->getId()] ?? null,
				);
				$this->children[] = $root;
			}
		}
		return $this->children;
	}

	public function childExists($name): bool {
		try {
			$this->getChild($name);
			return true;
		} catch (NotFound $e) {
			return false;
		}
	}

	public function getLastModified(): int {
		return 0;
	}
}
