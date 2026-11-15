# Security policy

## Reporting a vulnerability

Please report security problems privately. Do not open a public issue.

Use GitHub's private vulnerability reporting: open the **Security** tab of this
repository and choose **Report a vulnerability**. Include what you found, the
steps to reproduce it, and the impact you expect.

You can expect an acknowledgement within a few days. Please give us reasonable
time to fix the problem before disclosing it publicly.

## Scope

CodeAtlas is self-hosted. Reports about the application code in this repository
are in scope: authentication and sessions, authorization, the Slack integration,
the repository-reading agent, and the Docker setup. Problems in a deployment's
own configuration, such as a weak admin password or an exposed port, are the
operator's responsibility; the README lists the settings that harden a deployment.

## Supported versions

Only the latest commit on `main` receives security fixes.
