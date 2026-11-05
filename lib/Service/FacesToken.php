<?php

declare(strict_types=1);

/*
 * Copyright (c) 2026 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */

namespace OCA\Recognize\Service;

/**
 * Bearer token for python/api_server.py, which verifies it (verify_token):
 * b64url(json{"v":1,"uid","exp"}) "." b64url(HMAC-SHA256(secret, json)).
 *
 * No dependencies, so tools/check_token.php can load it without Nextcloud.
 */
final class FacesToken {
	/** The verifier rejects exp > now + 3600 (+60 s skew). */
	public const TTL = 3600;

	public static function mint(string $secret, string $uid, int $exp): string {
		$payload = json_encode(['v' => 1, 'uid' => $uid, 'exp' => $exp], JSON_THROW_ON_ERROR);
		return self::b64url($payload) . '.' . self::b64url(hash_hmac('sha256', $payload, $secret, true));
	}

	private static function b64url(string $data): string {
		return rtrim(strtr(base64_encode($data), '+/', '-_'), '=');
	}
}
