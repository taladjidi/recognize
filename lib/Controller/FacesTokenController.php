<?php

declare(strict_types=1);

/*
 * Copyright (c) 2026 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */

namespace OCA\Recognize\Controller;

use OCA\Recognize\Service\FacesToken;
use OCP\AppFramework\Controller;
use OCP\AppFramework\Http;
use OCP\AppFramework\Http\Attribute\NoAdminRequired;
use OCP\AppFramework\Http\JSONResponse;
use OCP\AppFramework\Services\IAppConfig;
use OCP\IRequest;
use OCP\IUserSession;

/**
 * Mints the bearer token the Photos frontend sends to the Faces read API
 * (python/api_server.py, behind nginx at /recognize-api/).
 *
 * Logged-in users only; CSRF is checked (Nextcloud's default), so the
 * frontend must send its requesttoken, which @nextcloud/axios does.
 */
final class FacesTokenController extends Controller {
	public function __construct(
		string $appName,
		IRequest $request,
		private IAppConfig $appConfig,
		private IUserSession $userSession,
	) {
		parent::__construct($appName, $request);
	}

	#[NoAdminRequired]
	public function token(): JSONResponse {
		$user = $this->userSession->getUser();
		// Sensitive values are decrypted by getAppValueString; written by python/setup_config.py.
		$secret = $this->appConfig->getAppValueString('api_token_secret', '');
		if ($user === null || $secret === '') {
			return new JSONResponse([], Http::STATUS_SERVICE_UNAVAILABLE);
		}
		$expires = time() + FacesToken::TTL;
		return new JSONResponse([
			'token' => FacesToken::mint($secret, $user->getUID(), $expires),
			'expires' => $expires,
		]);
	}
}
