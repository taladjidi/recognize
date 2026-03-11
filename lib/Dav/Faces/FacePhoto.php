<?php

/*
 * Copyright (c) 2022 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\Dav\Faces;

use OCA\Recognize\Db\FaceDetection;
use OCA\Recognize\Db\FaceDetectionMapper;
use OCP\Files\File;
use OCP\Files\Folder;
use OCP\IPreview;
use OCP\ITagManager;
use OCP\ITags;
use Override;
use Sabre\DAV\Exception\Forbidden;
use Sabre\DAV\Exception\NotFound;
use Sabre\DAV\IFile;

class FacePhoto implements IFile {
	private FaceDetectionMapper $detectionMapper;
	private FaceDetection $faceDetection;
	private Folder $userFolder;
	private ?File $file = null;
	private ITagManager $tagManager;
	private IPreview $preview;

	public function __construct(FaceDetectionMapper $detectionMapper, FaceDetection $faceDetection, Folder $userFolder, ITagManager $tagManager, IPreview $preview) {
		$this->detectionMapper = $detectionMapper;
		$this->faceDetection = $faceDetection;
		$this->userFolder = $userFolder;
		$this->tagManager = $tagManager;
		$this->preview = $preview;
	}

	#[Override]
	public function getName(): string {
		try {
			$file = $this->getFile();
			return $this->faceDetection->getId() . '-' . $file->getName();
		} catch (NotFound $e) {
			return $this->faceDetection->getId() . '-unknown';
		}
	}

	#[Override]
	public function delete(): void {
		$detections = $this->detectionMapper->findByClusterId($this->faceDetection->getClusterId());
		if (count($detections) > 1) {
			// Calculate centroid of all detections in the cluster
			$dim = 512;
			$sum = array_fill(0, $dim, 0.0);
			foreach ($detections as $det) {
				$vec = $det->getVector();
				for ($i = 0; $i < $dim; $i++) {
					$sum[$i] += $vec[$i];
				}
			}
			$n = (float)count($detections);
			$centroid = array_map(fn (float $v) => $v / $n, $sum);

			// Euclidean distance from this detection to centroid
			$thisVec = $this->faceDetection->getVector();
			$sqSum = 0.0;
			for ($i = 0; $i < $dim; $i++) {
				$diff = $centroid[$i] - $thisVec[$i];
				$sqSum += $diff * $diff;
			}
			$this->faceDetection->setThreshold(sqrt($sqSum));
		}
		$this->faceDetection->setClusterId(null);
		$this->detectionMapper->update($this->faceDetection);
	}

	#[Override]
	public function setName($name): never {
		throw new Forbidden('Cannot rename photos through faces API');
	}

	#[Override]
	public function put($data): never {
		throw new Forbidden('Can\'t write to photos trough the faces api');
	}

	public function getFile() : File {
		if ($this->file === null) {
			$node = $this->userFolder->getFirstNodeById($this->faceDetection->getFileId());
			if ($node !== null) {
				if ($node instanceof File) {
					return $this->file = $node;
				} else {
					throw new NotFound("Photo is a folder");
				}
			} else {
				throw new NotFound("Photo ".$this->faceDetection->getFileId()." not found for user");
			}
		} else {
			return $this->file;
		}
	}

	public function getFaceDetection() : FaceDetection {
		return $this->faceDetection;
	}

	/**
	 * @inheritDoc
	 * @throws \Sabre\DAV\Exception\NotFound
	 */
	public function get() {
		return $this->getFile()->fopen('r');
	}

	#[Override]
	public function getContentType(): string {
		return $this->getFile()->getMimeType();
	}

	#[Override]
	public function getETag(): string {
		return $this->getFile()->getEtag();
	}

	#[Override]
	public function getSize(): float|int {
		return $this->getFile()->getSize();
	}

	#[Override]
	public function getLastModified(): int {
		return $this->getFile()->getMTime();
	}

	public function hasPreview(): bool {
		return $this->preview->isAvailable($this->getFile());
	}

	public function isFavorite(): bool {
		$tagger = $this->tagManager->load('files');
		$tags = $tagger->getTagsForObjects([$this->getFile()->getId()]);

		if ($tags === false || empty($tags)) {
			return false;
		}

		return array_search(ITags::TAG_FAVORITE, current($tags)) !== false;
	}
}
