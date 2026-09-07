<?php

/*
 * Copyright (c) 2022 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\Dav\Faces;

use OCA\Recognize\Db\FaceCluster;
use OCA\Recognize\Db\FaceClusterMapper;
use OCA\Recognize\Db\FaceDetection;
use OCA\Recognize\Db\FaceDetectionMapper;
use OCP\AppFramework\Db\DoesNotExistException;
use OCP\AppFramework\Db\MultipleObjectsReturnedException;
use OCP\DB\Exception;
use OCP\Files\IRootFolder;
use OCP\IPreview;
use OCP\ITagManager;
use OCP\ITags;
use OCP\IUser;
use Sabre\DAV\Exception\Forbidden;
use Sabre\DAV\Exception\NotFound;
use Sabre\DAV\ICollection;
use Sabre\DAV\IMoveTarget;
use Sabre\DAV\INode;

final class FaceRoot implements ICollection, IMoveTarget {
	private FaceClusterMapper $clusterMapper;
	private FaceCluster $cluster;
	private IUser $user;
	private FaceDetectionMapper $detectionMapper;
	private IRootFolder $rootFolder;
	private ITagManager $tagManager;
	private IPreview $previewManager;
	/**
	 * @var \OCA\Recognize\Dav\Faces\FacePhoto[]
	 */
	private array $children = [];
	private ?int $preloadedCount = null;
	private ?FaceDetection $preloadedPreview = null;

	public function __construct(FaceClusterMapper $clusterMapper, FaceCluster $cluster, IUser $user, FaceDetectionMapper $detectionMapper, IRootFolder $rootFolder, ITagManager $tagManager, IPreview $previewManager) {
		$this->clusterMapper = $clusterMapper;
		$this->cluster = $cluster;
		$this->user = $user;
		$this->detectionMapper = $detectionMapper;
		$this->rootFolder = $rootFolder;
		$this->tagManager = $tagManager;
		$this->previewManager = $previewManager;
	}

	/** Inject pre-loaded metadata from batch queries to avoid per-cluster DB calls. */
	public function setPreloadedData(?int $count, ?FaceDetection $preview): void {
		$this->preloadedCount = $count;
		$this->preloadedPreview = $preview;
	}

	/**
	 * @inheritDoc
	 */
	public function getName(): string {
		return $this->cluster->getTitle() !== '' ? $this->cluster->getTitle() : ''.$this->cluster->getId();
	}

	/**
	 * @inheritDoc
	 */
	public function setName($name): void {
		try {
			$this->clusterMapper->findByUserAndTitle($this->user->getUID(), basename($name));
			throw new Forbidden('Not allowed to create duplicate names');
		} catch (DoesNotExistException $e) {
			// pass
		} catch (MultipleObjectsReturnedException|Exception $e) {
			throw $e;
		}
		$this->cluster->setTitle(basename($name));
		$this->clusterMapper->update($this->cluster);
	}

	/**
	 * @inheritDoc
	 */
	public function createDirectory($name): never {
		throw new Forbidden('Not allowed to create directories in this folder');
	}

	/**
	 * @inheritDoc
	 */
	public function createFile($name, $data = null): never {
		throw new Forbidden('Not allowed to create files in this folder');
	}

	/**
	 * @throws \OCP\Files\NotPermittedException
	 */
	public function getChildren(): array {
		if (count($this->children) !== 0) {
			return $this->children;
		}

		$detections = $this->detectionMapper->findByClusterIdWithExistingFiles($this->cluster->getId(), $this->user->getUID());
		$userFolder = $this->rootFolder->getUserFolder($this->user->getUID());

		// A PROPFIND emits the same properties for every child, so anything
		// fetched per photo becomes one query per photo. Fetch both in bulk
		// here instead. Measured on a 1331-photo cluster: favourites
		// 401 ms -> 1.9 ms, detections 314 ms -> 25 ms.
		$fileIds = array_values(array_unique(array_map(
			static fn (FaceDetection $detection) => $detection->getFileId(),
			$detections,
		)));

		// NOTE: getTagsForObjects() only returns entries for files that HAVE
		// tags, so an empty result means "nothing is favourited", not "the
		// lookup failed". Track success separately or every photo falls back
		// to its own query and the batching achieves nothing.
		$favorites = [];
		$favoritesLoaded = false;
		try {
			$tags = $this->tagManager->load('files')->getTagsForObjects($fileIds);
			if (is_array($tags)) {
				$favoritesLoaded = true;
				foreach ($tags as $fileId => $fileTags) {
					$favorites[(int)$fileId] = in_array(ITags::TAG_FAVORITE, $fileTags, true);
				}
			}
		} catch (\Throwable $e) {
			// Tags are cosmetic here; fall back to the per-photo lookup.
			$favoritesLoaded = false;
		}

		$detectionsByFile = [];
		$detectionsLoaded = false;
		try {
			$detectionsByFile = $this->detectionMapper->findByFileIdsWithTitle($fileIds);
			$detectionsLoaded = true;
		} catch (\Throwable $e) {
			$detectionsLoaded = false;
		}

		$this->children = array_map(function (FaceDetection $detection) use ($userFolder, $favorites, $favoritesLoaded, $detectionsByFile, $detectionsLoaded) {
			$photo = new FacePhoto($this->detectionMapper, $detection, $userFolder, $this->tagManager, $this->previewManager);
			$photo->setPreloadedData(
				$favoritesLoaded ? ($favorites[$detection->getFileId()] ?? false) : null,
				$detectionsLoaded ? ($detectionsByFile[$detection->getFileId()] ?? []) : null,
			);
			return $photo;
		}, $detections);

		return $this->children;
	}

	public function getChild($name): FacePhoto {
		if (count($this->children) !== 0) {
			foreach ($this->getChildren() as $child) {
				if ($child->getName() === $name) {
					return $child;
				}
			}
			throw new NotFound("$name not found");
		}

		// Try parsing detection ID from "{detectionId}-{filename}" format
		[$detectionId,] = explode('-', $name);
		if ((int)$detectionId > 0) {
			try {
				$detection = $this->detectionMapper->find((int)$detectionId);
				if ($detection->getClusterId() === $this->cluster->getId() && $detection->getUserId() === $this->user->getUID()) {
					return new FacePhoto($this->detectionMapper, $detection, $this->rootFolder->getUserFolder($this->user->getUID()), $this->tagManager, $this->previewManager);
				}
			} catch (DoesNotExistException $e) {
				// fall through to filename lookup
			}
		}

		// Fallback: Photos app sends real filenames (without detection ID prefix)
		// because FetchFacesMixin overwrites the DAV filename with realpath
		try {
			$detection = $this->detectionMapper->findByClusterIdAndFileName($this->cluster->getId(), $this->user->getUID(), $name);
			return new FacePhoto($this->detectionMapper, $detection, $this->rootFolder->getUserFolder($this->user->getUID()), $this->tagManager, $this->previewManager);
		} catch (DoesNotExistException $e) {
			throw new NotFound();
		}
	}

	public function childExists($name): bool {
		try {
			$this->getChild($name);
			return true;
		} catch (NotFound $e) {
			return false;
		}
	}

	public function moveInto($targetName, $sourcePath, INode $sourceNode) {
		if ($sourceNode instanceof FacePhoto) {
			$sourceNode->getFaceDetection()->setClusterId($this->cluster->getId());
			$this->detectionMapper->update($sourceNode->getFaceDetection());
			return true;
		}
		throw new Forbidden('Not a photo with a detected face, you can only move photos from the faces collection or the unassigned-faces here');
	}

	/**
	 * @inheritDoc
	 * @throws \OCP\DB\Exception
	 */
	public function delete(): void {
		$this->clusterMapper->delete($this->cluster);
	}

	public function getDetectionCount() : int {
		if ($this->preloadedCount !== null) {
			return $this->preloadedCount;
		}
		return $this->detectionMapper->countByClusterId($this->cluster->getId(), $this->user->getUID());
	}

	/**
	 * @inheritDoc
	 */
	public function getLastModified() : int {
		return 0;
	}

	public function getPreviewImage() : string {
		$detection = $this->preloadedPreview;
		if ($detection === null) {
			try {
				$detection = $this->detectionMapper->findDetectionForPreviewImageByClusterId($this->cluster->getId(), $this->user->getUID());
			} catch (DoesNotExistException|\OCP\DB\Exception $e) {
				return json_encode(null);
			}
		}
		return json_encode([
			'fileid' => $detection->getFileId(),
			'detection' => $detection->toArray()
		], \JSON_THROW_ON_ERROR);
	}
}
