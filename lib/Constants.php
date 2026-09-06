<?php

/*
 * Copyright (c) 2022 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize;

final class Constants {
	public const IMAGE_FORMATS = ['image/jpeg', 'image/png', 'image/bmp', 'image/heic', 'image/heif', 'image/webp'];
	public const AUDIO_FORMATS = ['audio/mpeg', 'audio/mp4', 'audio/ogg', 'audio/vnd.wav', 'audio/flac'];
	public const VIDEO_FORMATS = ['image/gif', 'video/mp4', 'video/MP2T', 'video/x-msvideo', 'video/x-ms-wmv', 'video/quicktime', 'video/ogg', 'video/mpeg', 'video/webm', 'video/x-matroska'];
	public const DIRECTORY_FORMATS = ['httpd/unix-directory'];
	public const IGNORE_MARKERS_ALL = ['.nomedia'];
	public const IGNORE_MARKERS_IMAGE = ['.noimage'];
	public const IGNORE_MARKERS_VIDEO = ['.novideo'];
	public const IGNORE_MARKERS_AUDIO = ['.nomusic'];
	public const MAX_FILE_SIZE = 10000000;

	/**
	 * Model identifiers.
	 *
	 * These are persisted values: they key the `<model>.enabled` entries in
	 * oc_appconfig and the `model` column of the recognize queue, so they must
	 * not be renamed. They previously lived as MODEL_NAME on the TensorFlow
	 * classifier classes; classification moved to the Python ONNX service, so
	 * the constants outlived the classes that defined them.
	 */
	public const MODEL_FACES = 'faces';
	public const MODEL_IMAGENET = 'imagenet';
	public const MODEL_LANDMARKS = 'landmarks';
	public const MODEL_MOVINET = 'movinet';
	public const MODEL_MUSICNN = 'musicnn';

	public const MODELS = [
		self::MODEL_FACES,
		self::MODEL_IMAGENET,
		self::MODEL_LANDMARKS,
		self::MODEL_MOVINET,
		self::MODEL_MUSICNN,
	];

	/**
	 * Background job classes from the TensorFlow era, removed when
	 * classification moved to the Python ONNX service.
	 *
	 * Existing installations can still hold oc_jobs rows for these, so the
	 * cleanup paths keep purging them by name. They are plain strings, not
	 * ::class references, precisely because the classes no longer exist.
	 */
	public const LEGACY_JOB_CLASSES = [
		'OCA\\Recognize\\BackgroundJobs\\ClassifyFacesJob',
		'OCA\\Recognize\\BackgroundJobs\\ClassifyImagenetJob',
		'OCA\\Recognize\\BackgroundJobs\\ClassifyLandmarksJob',
		'OCA\\Recognize\\BackgroundJobs\\ClassifyMovinetJob',
		'OCA\\Recognize\\BackgroundJobs\\ClassifyMusicnnJob',
		'OCA\\Recognize\\BackgroundJobs\\SchedulerJob',
		'OCA\\Recognize\\BackgroundJobs\\StorageCrawlJob',
		'OCA\\Recognize\\BackgroundJobs\\ProcessFsActionsJob',
	];

	/** Tags that must be present before landmark detection is attempted. */
	public const LANDMARKS_PRECONDITION_TAGS = ['Architecture', 'Tower', 'Monument', 'Bridge', 'Historic'];
}
