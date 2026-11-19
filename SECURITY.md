# Security and privacy

BioGuard is an academic biometric prototype. It is not a production access-control system and has not been evaluated for anti-spoofing, presentation attacks, secure enclave storage, or operational threat models.

## Never commit

Do not commit or publish:

- `flask_game/private_data/` and any user registry;
- face photographs, voice recordings, fingerprint uploads, or derived templates from real users;
- admin PINs, password hashes, session secrets, API keys, `.env` files, or tokens;
- audit logs, diagnostic dumps, temporary enrollment files, or backups;
- virtual environments, raw datasets, or local databases.

The `.gitignore` is a guardrail, not a privacy guarantee. Check `git status`, staged paths, and the complete history before every public push. If a secret or biometric sample is ever committed, revoke/rotate the secret, preserve evidence, remove the content from all reachable history, and assess whether the repository must be replaced.

## Serialized model warning

`joblib` and pickle-based artifacts can execute code during deserialization. Only load model files from this reviewed repository or another trusted source. Do not replace them with an untrusted download.

## Local runtime boundary

The Flask application creates its admin configuration, session secret, audit log, enrollment gallery, user folders, and temporary staging files below the ignored private root. Keep the default host on loopback for a classroom demo. If using a tunnel, treat the URL as public and review authentication, consent, browser permissions, rate limits, and data retention first.

## Reporting a problem

Do not post private biometric material, credentials, or exploit details in a public issue. For a suspected exposure, stop sharing the repository, revoke affected credentials, and contact the repository owner privately through an authenticated GitHub channel.
