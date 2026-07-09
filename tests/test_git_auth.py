import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.repos import branches
from app.repos.cloning import clone_repo
from app.repos.git_auth import (
    GitAuthError,
    git_env_for_repo,
    git_env_for_url,
    sanitize_git_error,
    validate_clone_url,
)


class GitAuthTests(unittest.TestCase):
    def test_github_https_uses_only_github_token(self):
        with patch.dict(
            os.environ,
            {
                "GH_TOKEN": "github-secret",
                "CODEATLAS_BITBUCKET_API_TOKEN": "bitbucket-secret",
            },
            clear=True,
        ):
            env = git_env_for_url("https://github.com/example/private.git")

        self.assertEqual(env["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(env["GIT_ASKPASS"].split("/")[-1], "git_askpass.py")
        self.assertEqual(env["CODEATLAS_GIT_ASKPASS_USERNAME"], "x-access-token")
        self.assertEqual(env["CODEATLAS_GIT_ASKPASS_TOKEN"], "github-secret")
        self.assertNotIn("bitbucket-secret", env.values())
        self.assertNotIn("GH_TOKEN", env)

    def test_bitbucket_https_uses_only_bitbucket_token(self):
        with patch.dict(
            os.environ,
            {
                "GH_TOKEN": "github-secret",
                "CODEATLAS_BITBUCKET_API_TOKEN": "bitbucket-secret",
            },
            clear=True,
        ):
            env = git_env_for_url("https://bitbucket.org/workspace/private.git")

        self.assertEqual(
            env["CODEATLAS_GIT_ASKPASS_USERNAME"],
            "x-bitbucket-api-token-auth",
        )
        self.assertEqual(env["CODEATLAS_GIT_ASKPASS_TOKEN"], "bitbucket-secret")
        self.assertNotIn("github-secret", env.values())

    def test_credentials_are_not_sent_to_arbitrary_hosts(self):
        with patch.dict(
            os.environ,
            {
                "GH_TOKEN": "github-secret",
                "CODEATLAS_BITBUCKET_API_TOKEN": "bitbucket-secret",
            },
            clear=True,
        ):
            env = git_env_for_url("https://git.example.internal/org/repo.git")

        self.assertEqual(env["GIT_TERMINAL_PROMPT"], "0")
        self.assertNotIn("GIT_ASKPASS", env)
        self.assertNotIn("CODEATLAS_GIT_ASKPASS_TOKEN", env)
        self.assertNotIn("github-secret", env.values())
        self.assertNotIn("bitbucket-secret", env.values())

    def test_clone_url_validation_blocks_embedded_credentials(self):
        with self.assertRaises(GitAuthError):
            validate_clone_url(
                "https://user:token@github.com/example/private.git",
                "https",
            )
        with self.assertRaises(GitAuthError):
            validate_clone_url(
                "https://bitbucket.org/workspace/private.git?access_token=secret",
                "https",
            )

    def test_clone_url_validation_allows_public_https_and_ssh(self):
        validate_clone_url("https://github.com/example/public.git", "https")
        validate_clone_url("git@github.com:example/private.git", "ssh")
        validate_clone_url("owner/repo", "gh")

    def test_git_error_redacts_configured_tokens_and_url_credentials(self):
        with patch.dict(
            os.environ,
            {
                "GH_TOKEN": "github-secret",
                "CODEATLAS_BITBUCKET_API_TOKEN": "bitbucket-secret",
            },
            clear=True,
        ):
            message = sanitize_git_error(
                "fatal https://user:github-secret@github.com/org/repo.git "
                "and bitbucket-secret"
            )

        self.assertNotIn("github-secret", message)
        self.assertNotIn("bitbucket-secret", message)
        self.assertIn("[redacted]", message)

    def test_clone_repo_passes_provider_auth_without_token_in_command(self):
        completed = subprocess.CompletedProcess(
            ["git"],
            0,
            stdout="",
            stderr="",
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"GH_TOKEN": "github-secret"},
            clear=True,
        ), patch(
            "app.repos.cloning.repo_clone_dir",
            return_value=Path(temp_dir) / "repo",
        ), patch(
            "app.repos.cloning.subprocess.run",
            return_value=completed,
        ) as run:
            clone_repo("https://github.com/example/private.git", "https", "sample")

        args = run.call_args.args[0]
        env = run.call_args.kwargs["env"]
        self.assertEqual(args[:3], ["git", "clone", "--depth"])
        self.assertNotIn("github-secret", args)
        self.assertEqual(env["CODEATLAS_GIT_ASKPASS_TOKEN"], "github-secret")

    def test_branch_git_passes_auth_environment(self):
        completed = subprocess.CompletedProcess(
            ["git"],
            0,
            stdout="",
            stderr="",
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "app.repos.branches.git_env_for_repo",
            return_value={"PATH": "/usr/bin", "CODEATLAS_GIT_ASKPASS_TOKEN": "secret"},
        ), patch(
            "app.repos.branches.subprocess.run",
            return_value=completed,
        ) as run:
            branches._git(Path(temp_dir), "fetch", "origin", check=False)

        self.assertEqual(
            run.call_args.kwargs["env"]["CODEATLAS_GIT_ASKPASS_TOKEN"],
            "secret",
        )

    def test_repo_origin_with_embedded_credentials_is_rejected(self):
        with patch(
            "app.repos.git_auth.origin_url_for_repo",
            return_value="https://user:secret@github.com/org/private.git",
        ):
            with self.assertRaises(GitAuthError):
                git_env_for_repo(Path("/tmp/repo"))
