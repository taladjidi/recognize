<?php

/*
 * Copyright (c) 2021-2022 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\AppInfo;

use OCA\DAV\Connector\Sabre\Principal;
use OCA\DAV\Events\SabrePluginAddEvent;
use OCA\Recognize\Dav\Faces\PropFindPlugin;
use OCA\Recognize\Hooks\FileListener;
use OCP\AppFramework\App;
use OCP\AppFramework\Bootstrap\IBootContext;
use OCP\AppFramework\Bootstrap\IBootstrap;
use OCP\AppFramework\Bootstrap\IRegistrationContext;
use OCP\EventDispatcher\IEventDispatcher;
use OCP\Files\Events\Node\BeforeNodeDeletedEvent;
use OCP\Files\Events\Node\NodeCreatedEvent;
use OCP\Files\Events\Node\NodeRenamedEvent;

final class Application extends App implements IBootstrap {
	public const APP_ID = 'recognize';

	public function __construct() {
		parent::__construct(self::APP_ID);
		/**
		 * @var IEventDispatcher $dispatcher
		 */
		$dispatcher = $this->getContainer()->get(IEventDispatcher::class);
		$dispatcher->addServiceListener(NodeCreatedEvent::class, FileListener::class);
		$dispatcher->addServiceListener(NodeRenamedEvent::class, FileListener::class);
		$dispatcher->addServiceListener(BeforeNodeDeletedEvent::class, FileListener::class);
	}

	public function register(IRegistrationContext $context): void {
		@include_once __DIR__ . '/../../vendor/scoper-autoload.php';
		@include_once __DIR__ . '/../../vendor/rubix/ml/src/functions.php';
		@include_once __DIR__ . '/../../vendor/rubix/ml/src/constants.php';

		/** Register $principalBackend for the DAV collection */
		$context->registerServiceAlias('principalBackend', Principal::class);
	}

	/**
	 * @throws \Throwable
	 */
	public function boot(IBootContext $context): void {
		$eventDispatcher = \OCP\Server::get(IEventDispatcher::class);
		$eventDispatcher->addListener(SabrePluginAddEvent::class, function (SabrePluginAddEvent $event): void {
			$server = $event->getServer();

			// We have to register the PropFindPlugin here and not info.xml,
			// because info.xml plugins are loaded, after the
			// beforeMethod:* hook has already been emitted.
			$server->addPlugin($this->getContainer()->get(PropFindPlugin::class));
		});
	}
}
