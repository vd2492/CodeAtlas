import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException, Response

from app import db, main
from app.auth import routes as auth_routes
from app.auth.security import hash_password


def fake_request(host="203.0.113.7"):
    return SimpleNamespace(client=SimpleNamespace(host=host))


class SecurityHeaderTests(unittest.TestCase):
    @staticmethod
    def run_middleware():
        async def call_next(request):
            return Response("ok")

        return asyncio.run(main.add_noindex_header(SimpleNamespace(), call_next))

    def test_pages_cannot_be_framed_or_sniffed(self):
        headers = self.run_middleware().headers
        self.assertEqual(headers["x-frame-options"], "DENY")
        self.assertEqual(headers["x-content-type-options"], "nosniff")
        self.assertEqual(headers["referrer-policy"], "same-origin")
        # The existing header is still set.
        self.assertIn("noindex", headers["x-robots-tag"])

    def test_hsts_only_when_secure_cookies_are_on(self):
        with patch.object(main, "COOKIE_SECURE", True):
            self.assertIn("max-age", self.run_middleware().headers["strict-transport-security"])
        with patch.object(main, "COOKIE_SECURE", False):
            self.assertNotIn("strict-transport-security", self.run_middleware().headers)

    def test_a_route_can_still_set_its_own_value(self):
        async def call_next(request):
            return Response("ok", headers={"X-Frame-Options": "SAMEORIGIN"})

        response = asyncio.run(main.add_noindex_header(SimpleNamespace(), call_next))
        self.assertEqual(response.headers["x-frame-options"], "SAMEORIGIN")


class LoginHardeningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(db, "DB_PATH", Path(self.temp.name) / "codeatlas.db")
        self.db_patch.start()
        db.init_db()
        db.create_user("reader", hash_password("right-password"))
        auth_routes._login_failures.clear()
        auth_routes._login_address_failures.clear()

    def tearDown(self):
        auth_routes._login_failures.clear()
        auth_routes._login_address_failures.clear()
        self.db_patch.stop()
        self.temp.cleanup()

    def attempt(self, username, password, host="203.0.113.7"):
        return auth_routes.login(
            auth_routes.Credentials(username=username, password=password),
            Response(),
            fake_request(host),
        )

    def failure_status(self, username, password, host="203.0.113.7"):
        with self.assertRaises(HTTPException) as raised:
            self.attempt(username, password, host)
        return raised.exception.status_code

    def test_one_address_cannot_spray_many_usernames(self):
        with patch.object(auth_routes, "LOGIN_ADDRESS_RATE_LIMIT", 3):
            for index in range(3):
                self.assertEqual(self.failure_status(f"user-{index}", "guess"), 401)
            # A fourth, different username is refused before any password check.
            self.assertEqual(self.failure_status("user-9", "guess"), 429)
            self.assertEqual(self.failure_status("reader", "right-password"), 429)
            # Another address is unaffected.
            self.assertEqual(
                self.attempt("reader", "right-password", host="198.51.100.9")["user"]["username"],
                "reader",
            )

    def test_address_failures_expire(self):
        with patch.object(auth_routes, "LOGIN_ADDRESS_RATE_LIMIT", 2), patch.object(
            auth_routes, "LOGIN_RATE_WINDOW_SECONDS", 300
        ):
            for index in range(2):
                self.failure_status(f"user-{index}", "guess")
            self.assertEqual(self.failure_status("user-9", "guess"), 429)
            with patch.object(auth_routes.time, "monotonic", return_value=10_000_000.0):
                self.assertEqual(self.failure_status("user-9", "guess"), 401)

    def test_per_username_limit_and_success_still_work_as_before(self):
        with patch.object(auth_routes, "LOGIN_RATE_LIMIT", 2):
            for _ in range(2):
                self.assertEqual(self.failure_status("reader", "wrong"), 401)
            self.assertEqual(self.failure_status("reader", "right-password"), 429)
        auth_routes._login_failures.clear()
        self.assertEqual(self.attempt("reader", "right-password")["user"]["username"], "reader")

    def test_login_without_a_request_object_still_works(self):
        result = auth_routes.login(
            auth_routes.Credentials(username="reader", password="right-password"),
            Response(),
        )
        self.assertEqual(result["user"]["username"], "reader")

    def test_unknown_and_known_users_do_the_same_work_and_get_the_same_error(self):
        calls = []
        real = auth_routes.verify_password

        def counting(password, encoded):
            calls.append(encoded)
            return real(password, encoded)

        with patch.object(auth_routes, "verify_password", side_effect=counting):
            with self.assertRaises(HTTPException) as unknown:
                self.attempt("nobody", "guess")
            unknown_calls = len(calls)
            with self.assertRaises(HTTPException) as known:
                self.attempt("reader", "guess")
        self.assertEqual(unknown_calls, 1)
        self.assertEqual(len(calls) - unknown_calls, 1)
        self.assertEqual(unknown.exception.status_code, known.exception.status_code)
        self.assertEqual(unknown.exception.detail, known.exception.detail)

    def test_accounts_without_a_password_still_fail_with_equal_work(self):
        db.create_google_user("g@example.com", google_sub="sub-1")
        with patch.object(
            auth_routes, "verify_password", wraps=auth_routes.verify_password
        ) as spy:
            self.assertEqual(self.failure_status("g@example.com", "guess"), 401)
        self.assertEqual(spy.call_count, 1)


if __name__ == "__main__":
    unittest.main()
