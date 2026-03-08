<?php

/*
 * Copyright (c) 2021-2022 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\Classifiers;

use OCA\Recognize\Classifiers\Images\ClusteringFaceClassifier;
use OCA\Recognize\Classifiers\Images\ImagenetClassifier;
use OCA\Recognize\Classifiers\Images\LandmarksClassifier;
use OCA\Recognize\Constants;
use OCA\Recognize\Db\QueueFile;
use OCA\Recognize\Service\QueueService;
use OCP\AppFramework\Services\IAppConfig;
use OCP\DB\Exception;
use OCP\Encryption\Exceptions\GenericEncryptionException;
use OCP\Files\File;
use OCP\Files\InvalidPathException;
use OCP\Files\IRootFolder;
use OCP\Files\Node;
use OCP\Files\NotFoundException;
use OCP\Files\NotPermittedException;
use OCP\IConfig;
use OCP\IPreview;
use OCP\ITempManager;
use Psr\Log\LoggerInterface;

abstract class Classifier {
	public const TEMP_FILE_DIMENSION = 1024;
	public const MAX_EXECUTION_TIME = 0;

	protected LoggerInterface $logger;
	protected IAppConfig $config;
	protected IRootFolder $rootFolder;
	protected QueueService $queue;
	private ITempManager $tempManager;
	private IPreview $previewProvider;
	private int $maxExecutionTime = self::MAX_EXECUTION_TIME;

	public function __construct(LoggerInterface $logger, IAppConfig $config, IRootFolder $rootFolder, QueueService $queue, ITempManager $tempManager, IPreview  $previewProvider) {
		$this->logger = $logger;
		$this->config = $config;
		$this->rootFolder = $rootFolder;
		$this->queue = $queue;
		$this->tempManager = $tempManager;
		$this->previewProvider = $previewProvider;
	}

	public function setMaxExecutionTime(int $time): void {
		$this->maxExecutionTime = $time;
	}

	/**
	 * @param list<QueueFile> $queueFiles
	 * @throws \ErrorException|\RuntimeException
	 */
	abstract public function classify(array $queueFiles): void;

	/**
	 * Build the command and environment for the classifier subprocess.
	 *
	 * @param string $model
	 * @return array{command: list<string>, env: array<string,string>}
	 */
	private function buildCommand(string $model): array {
		$pythonBinary = $this->config->getAppValueString('python_binary', '');
		if ($pythonBinary !== '') {
			$command = [
				$pythonBinary,
				dirname(__DIR__, 2) . '/python/classifier_'.$model.'.py',
				'-'
			];
		} else {
			$command = [
				$this->config->getAppValueString('node_binary'),
				dirname(__DIR__, 2) . '/src/classifier_'.$model.'.js',
				'-'
			];
		}

		if (trim($this->config->getAppValueString('nice_binary', '')) !== '') {
			$command = [
				$this->config->getAppValueString('nice_binary'),
				"-" . $this->config->getAppValueString('nice_value', '0'),
				...$command,
			];
		}

		$env = [];
		if ($this->config->getAppValueString('tensorflow.gpu', 'false') === 'true') {
			$env['RECOGNIZE_GPU'] = 'true';
		}
		if ($this->config->getAppValueString('tensorflow.purejs', 'false') === 'true') {
			$env['RECOGNIZE_PUREJS'] = 'true';
		}
		$cores = $this->config->getAppValueString('tensorflow.cores', '0');
		if ($cores !== '0') {
			$env['RECOGNIZE_CORES'] = $cores;
		}
		$ffmpegBinary = $this->config->getAppValueString('ffmpeg_binary', '');
		if ($ffmpegBinary !== '') {
			$env['FFMPEG_BINARY'] = $ffmpegBinary;
		}
		$imagenetModel = $this->config->getAppValueString('imagenet.model', 'auto');
		if ($imagenetModel !== 'auto') {
			$env['RECOGNIZE_IMAGENET_MODEL'] = $imagenetModel;
		}

		return ['command' => $command, 'env' => $env];
	}

	/**
	 * Resolve a QueueFile to a local filesystem path, or null if it should be skipped.
	 *
	 * @param string $model
	 * @param QueueFile $queueFile
	 * @return array{path: string, name: string}|null
	 */
	private function resolveFile(string $model, QueueFile $queueFile): ?array {
		$file = $this->rootFolder->getFirstNodeById($queueFile->getFileId());
		if ($file === null) {
			try {
				$this->logger->debug('removing '.$queueFile->getFileId().' from '.$model.' queue because it couldn\'t be found');
				$this->queue->removeFromQueue($model, $queueFile);
			} catch (Exception $e) {
				$this->logger->warning($e->getMessage(), ['exception' => $e]);
			}
			return null;
		}
		try {
			if ($file->getSize() == 0) {
				$this->logger->debug('File is empty: ' . $file->getPath());
				try {
					$this->queue->removeFromQueue($model, $queueFile);
				} catch (Exception $e) {
					$this->logger->warning($e->getMessage(), ['exception' => $e]);
				}
				return null;
			}
			$path = $this->getConvertedFilePath($file);
			if (in_array($model, [ImagenetClassifier::MODEL_NAME, LandmarksClassifier::MODEL_NAME, ClusteringFaceClassifier::MODEL_NAME], true)) {
				$filesize = filesize($path);
				if ($filesize !== false && $filesize / (1024 * 1024) > 50) {
					$this->logger->debug('File is too large for classifier: ' . $file->getPath());
					try {
						$this->queue->removeFromQueue($model, $queueFile);
					} catch (Exception $e) {
						$this->logger->warning($e->getMessage(), ['exception' => $e]);
					}
					return null;
				}
			}
			return ['path' => $path, 'name' => $file->getPath()];
		} catch (NotFoundException|InvalidPathException $e) {
			$this->logger->warning('Could not find file', ['exception' => $e]);
			try {
				$this->queue->removeFromQueue($model, $queueFile);
			} catch (Exception $e) {
				$this->logger->warning($e->getMessage(), ['exception' => $e]);
			}
			return null;
		} catch (GenericEncryptionException $e) {
			$this->logger->warning('Could not load encrypted file', ['exception' => $e]);
			try {
				$this->queue->removeFromQueue($model, $queueFile);
			} catch (Exception $e) {
				$this->logger->warning($e->getMessage(), ['exception' => $e]);
			}
			return null;
		}
	}

	/**
	 * Streaming producer/consumer classifier pipeline.
	 *
	 * Spawns the classifier subprocess once, then continuously:
	 *   - PRODUCER: resolves the next file to a local path and writes it to stdin
	 *   - CONSUMER: reads JSON results from stdout and yields them
	 *
	 * File resolution (DB lookup + filesystem) overlaps with inference on the GPU,
	 * keeping the GPU saturated instead of idle between batches.
	 *
	 * @param string $model
	 * @param list<QueueFile> $queueFiles
	 * @param int $timeout Seconds per file
	 * @return \Generator
	 * @psalm-return \Generator<QueueFile, mixed, mixed, null>
	 * @throws \ErrorException|\RuntimeException
	 */
	public function classifyFiles(string $model, array $queueFiles, int $timeout): \Generator {
		if (count($queueFiles) === 0) {
			$this->logger->debug('No files left to classify');
			return;
		}

		$startTime = time();
		$built = $this->buildCommand($model);
		$command = $built['command'];
		$env = $built['env'];

		$this->logger->debug('Running '.var_export($command, true));
		$this->logger->info('Classifying ' . count($queueFiles) . ' files with ' . $model);

		// Use proc_open for full control over stdin/stdout pipes.
		// This lets us write paths incrementally while reading results,
		// overlapping PHP file resolution with Python GPU inference.
		$descriptors = [
			0 => ['pipe', 'r'],  // stdin: child reads, parent writes
			1 => ['pipe', 'w'],  // stdout: child writes, parent reads
			2 => ['pipe', 'w'],  // stderr: child writes, parent reads
		];

		// Merge env with current environment
		$procEnv = array_merge(getenv(), $env);

		$proc = proc_open($command, $descriptors, $pipes, __DIR__, $procEnv);
		if (!is_resource($proc)) {
			throw new \ErrorException('Classifier process could not be started');
		}

		$stdin = $pipes[0];
		$stdout = $pipes[1];
		$stderr = $pipes[2];

		// Make stdout and stderr non-blocking so we can poll them
		stream_set_blocking($stdout, false);
		stream_set_blocking($stderr, false);

		// Set CPU affinity if cores are configured
		$cores = $this->config->getAppValueString('tensorflow.cores', '0');
		if ((int)$cores !== 0) {
			$status = proc_get_status($proc);
			if ($status['running']) {
				@exec('taskset -cp ' . implode(',', range(0, (int)$cores, 1)) . ' ' . ((string)$status['pid']));
			}
		}

		// Pipeline state
		$sentFiles = [];      // QueueFiles we've sent to the process (in order)
		$sentPaths = [];      // Corresponding paths (for logging)
		$sentNames = [];      // Corresponding display names (for logging)
		$sendIndex = 0;       // Next queueFile to resolve and send
		$recvIndex = 0;       // Next result to read
		$buffer = '';
		$errOut = '';
		$stdinOpen = true;

		// Pre-resolve a window of files ahead of the process to keep it fed.
		// We'll resolve and send files as fast as possible, then drain results.
		$totalFiles = count($queueFiles);

		// How many files to keep "in flight" (sent but not yet received)
		// This controls how far ahead the producer runs vs the consumer.
		$prefetchWindow = 8;

		while ($recvIndex < count($sentFiles) || $sendIndex < $totalFiles) {
			// Check max execution time
			if ($this->maxExecutionTime > 0 && time() - $startTime > $this->maxExecutionTime) {
				break;
			}

			// PRODUCER: resolve and send files while we have room in the preflight window
			while ($stdinOpen && $sendIndex < $totalFiles && (count($sentFiles) - $recvIndex) < $prefetchWindow) {
				if ($this->maxExecutionTime > 0 && time() - $startTime > $this->maxExecutionTime) {
					break;
				}

				$queueFile = $queueFiles[$sendIndex];
				$sendIndex++;

				$resolved = $this->resolveFile($model, $queueFile);
				if ($resolved === null) {
					continue; // Skip this file, don't send to process
				}

				$line = $resolved['path'] . "\n";
				$written = @fwrite($stdin, $line);
				if ($written === false) {
					$this->logger->warning('Failed to write to classifier stdin');
					break;
				}
				fflush($stdin);

				$sentFiles[] = $queueFile;
				$sentPaths[] = $resolved['path'];
				$sentNames[] = $resolved['name'];
			}

			// Close stdin once all files have been sent
			if ($stdinOpen && $sendIndex >= $totalFiles && (count($sentFiles) - $recvIndex) <= count($sentFiles)) {
				// Only close once we've sent everything
				if ($sendIndex >= $totalFiles) {
					fclose($stdin);
					$stdinOpen = false;
				}
			}

			// Nothing was sent (all files were invalid)
			if (count($sentFiles) === 0 && $sendIndex >= $totalFiles) {
				if ($stdinOpen) {
					fclose($stdin);
					$stdinOpen = false;
				}
				break;
			}

			// CONSUMER: read results from stdout
			$readStreams = [$stdout, $stderr];
			$write = null;
			$except = null;
			$changed = @stream_select($readStreams, $write, $except, $timeout);

			if ($changed === false) {
				$this->logger->warning('stream_select failed');
				break;
			}

			// Read stderr (non-blocking)
			$errData = stream_get_contents($stderr);
			if ($errData !== false && $errData !== '') {
				$errOut .= $errData;
				$this->logger->debug('Classifier process output: ' . $errData);
			}

			// Read stdout
			$data = stream_get_contents($stdout);
			if ($data !== false && $data !== '') {
				$buffer .= $data;
				$lines = explode("\n", $buffer);
				$buffer = '';

				foreach ($lines as $result) {
					if (trim($result) === '') {
						continue;
					}
					// Validate JSON before processing
					try {
						json_decode($result, true, 512, JSON_OBJECT_AS_ARRAY | JSON_THROW_ON_ERROR | JSON_INVALID_UTF8_IGNORE);
						$valid = true;
					} catch (\JsonException $e) {
						$valid = false;
					}
					if (!$valid) {
						$buffer .= "\n" . $result;
						continue;
					}

					if ($recvIndex >= count($sentFiles)) {
						$this->logger->warning('Received more results than files sent');
						break;
					}

					$this->logger->debug('Result for ' . $sentNames[$recvIndex] . '(' . basename($sentPaths[$recvIndex]) . ') = ' . $result);
					try {
						$results = json_decode($result, true, 512, JSON_OBJECT_AS_ARRAY | JSON_THROW_ON_ERROR | JSON_INVALID_UTF8_IGNORE);
						yield $sentFiles[$recvIndex] => $results;
						$this->queue->removeFromQueue($model, $sentFiles[$recvIndex]);
					} catch (\JsonException $e) {
						$this->logger->warning('JSON exception');
						$this->logger->warning($e->getMessage(), ['exception' => $e]);
						$this->logger->warning($result);
					} catch (Exception $e) {
						$this->logger->warning($e->getMessage(), ['exception' => $e]);
					}
					$recvIndex++;
				}
			}

			// Check if process has exited
			$status = proc_get_status($proc);
			if (!$status['running'] && ($data === false || $data === '')) {
				// Drain any remaining stdout
				$remaining = stream_get_contents($stdout);
				if ($remaining !== false && $remaining !== '') {
					$buffer .= $remaining;
					// Process remaining buffer (same logic as above)
					$lines = explode("\n", $buffer);
					foreach ($lines as $result) {
						if (trim($result) === '' || $recvIndex >= count($sentFiles)) {
							continue;
						}
						try {
							$results = json_decode($result, true, 512, JSON_OBJECT_AS_ARRAY | JSON_THROW_ON_ERROR | JSON_INVALID_UTF8_IGNORE);
							$this->logger->debug('Result for ' . $sentNames[$recvIndex] . '(' . basename($sentPaths[$recvIndex]) . ') = ' . $result);
							yield $sentFiles[$recvIndex] => $results;
							$this->queue->removeFromQueue($model, $sentFiles[$recvIndex]);
							$recvIndex++;
						} catch (\JsonException $e) {
							continue;
						} catch (Exception $e) {
							$this->logger->warning($e->getMessage(), ['exception' => $e]);
							$recvIndex++;
						}
					}
				}
				break;
			}
		}

		// Cleanup
		if ($stdinOpen) {
			@fclose($stdin);
		}
		@fclose($stdout);

		// Drain stderr
		$errRemaining = stream_get_contents($stderr);
		if ($errRemaining !== false && $errRemaining !== '') {
			$errOut .= $errRemaining;
		}
		@fclose($stderr);

		$exitCode = proc_close($proc);
		$this->cleanUpTmpFiles();

		if ($recvIndex !== count($sentFiles)) {
			$this->logger->warning('Classifier process output (stderr): ' . substr($errOut, 0, 2000));
			$this->logger->warning('Expected ' . count($sentFiles) . ' results, got ' . $recvIndex . ', exit code: ' . $exitCode);
			throw new \ErrorException('Classifier process error');
		}
	}

	/**
	 * Get path of file to process.
	 *
	 * @param \OCP\Files\Node $file
	 * @return string Path to file to process
	 * @throws \OCP\Files\NotFoundException
	 * @throws GenericEncryptionException
	 */
	private function getConvertedFilePath(Node $file): string {
		if (!$file instanceof File) {
			throw new NotFoundException();
		}
		$path = $file->getStorage()->getLocalFile($file->getInternalPath());

		if (!is_string($path)) {
			throw new NotFoundException();
		}

		// Skip PHP preview generation — Python classifiers handle their own
		// resizing (PIL/numpy) which is faster than PHP GD/Imagick and avoids
		// the CPU bottleneck of generating intermediate preview files.
		return $path;
	}

	public function cleanUpTmpFiles():void {
		$this->tempManager->clean();
	}

	/**
	 * @param File $file
	 * @return string
	 * @throws \OCA\Recognize\Exception\Exception|NotFoundException
	 */
	public function generatePreviewWithProvider(File $file): string {
		$image = $this->previewProvider->getPreview($file, self::TEMP_FILE_DIMENSION, self::TEMP_FILE_DIMENSION);

		try {
			$preview = $image->read();
		} catch (NotPermittedException $e) {
			throw new \OCA\Recognize\Exception\Exception('Could not read preview file', 0, $e);
		}

		if ($preview === false) {
			throw new \OCA\Recognize\Exception\Exception('Could not open preview file');
		}

		// Create a temporary file *with the correct extension*
		$tmpname = $this->tempManager->getTemporaryFile('.jpg');

		if ($tmpname === false) {
			throw new \OCA\Recognize\Exception\Exception('Could not create tmpfile');
		}

		$tmpfile = fopen($tmpname, 'wb');

		if ($tmpfile === false) {
			throw new \OCA\Recognize\Exception\Exception('Could not open tmpfile');
		}

		$copyResult = stream_copy_to_stream($preview, $tmpfile);
		fclose($preview);
		fclose($tmpfile);

		if ($copyResult === false) {
			throw new \OCA\Recognize\Exception\Exception('Could not copy preview file to temp folder');
		}

		$imagetype = exif_imagetype($tmpname);

		if (in_array($imagetype, [IMAGETYPE_WEBP, IMAGETYPE_AVIF, false])) { // To troubleshoot if it is a webp or avif.
			$imageString = file_get_contents($tmpname);
			if ($imageString === false) {
				throw new \OCA\Recognize\Exception\Exception('Could not load preview file from temp folder');
			}
			$previewImage = imagecreatefromstring($imageString);
			if ($previewImage === false) {
				throw new \OCA\Recognize\Exception\Exception('Could not load preview file from temp folder');
			}
			$use_gd_quality = (int)\OCP\Server::get(IConfig::class)->getSystemValue('recognize.preview.quality', '100');
			if (imagejpeg($previewImage, $tmpname, $use_gd_quality) === false) {
				imagedestroy($previewImage);
				throw new \OCA\Recognize\Exception\Exception('Could not copy preview file to temp folder');
			}
			imagedestroy($previewImage);
		}

		return $tmpname;
	}

	/**
	 * @param string $path
	 * @return string
	 * @throws \OCA\Recognize\Exception\Exception
	 */
	public function generatePreviewWithGD(string $path): string {
		$imageContents = file_get_contents($path);
		if (!$imageContents) {
			throw new \OCA\Recognize\Exception\Exception('Could not load image for preview with gdlib');
		}
		$image = imagecreatefromstring($imageContents);
		if (!$image) {
			throw new \OCA\Recognize\Exception\Exception('Could not load image for preview with gdlib');
		}
		$width = imagesx($image);
		$height = imagesy($image);

		if ($width === false || $height === false) {
			throw new \OCA\Recognize\Exception\Exception('Could not get image dimensions for preview with gdlib');
		}

		$maxWidth = (float) self::TEMP_FILE_DIMENSION;
		$maxHeight = (float) self::TEMP_FILE_DIMENSION;

		if ($width > $maxWidth || $height > $maxHeight) {
			$aspectRatio = (float) ($width / $height);
			if ($width > $height) {
				$newWidth = $maxWidth;
				$newHeight = $maxWidth / $aspectRatio;
			} else {
				$newHeight = $maxHeight;
				$newWidth = $maxHeight * $aspectRatio;
			}
			$previewImage = imagescale($image, (int)$newWidth, (int)$newHeight);
		} else {
			return $path;
		}

		// Create a temporary file *with the correct extension*
		$tmpname = $this->tempManager->getTemporaryFile('.jpg');

		if ($tmpname === false) {
			throw new \OCA\Recognize\Exception\Exception('Could not create tmpfile');
		}

		$use_gd_quality = (int)\OCP\Server::get(IConfig::class)->getSystemValue('recognize.preview.quality', '100');
		if (imagejpeg($previewImage, $tmpname, $use_gd_quality) === false) {
			imagedestroy($image);
			imagedestroy($previewImage);
			throw new \OCA\Recognize\Exception\Exception('Could not copy preview file to temp folder');
		}
		imagedestroy($image);
		imagedestroy($previewImage);

		return $tmpname;
	}
}
