# Releasing

Releases are driven by git tags. The version comes from the tag
(setuptools-scm), and the release notes come from `CHANGELOG.md`.

## One-time setup

1. **Trusted publishers** (no API tokens are stored). On both
   [pypi.org](https://pypi.org/manage/project/tmgen/settings/publishing/) and
   [test.pypi.org](https://test.pypi.org/manage/project/tmgen/settings/publishing/)
   add a GitHub publisher: owner `progwriter`, repository `TMgen`, workflow
   `release.yml`, environment `pypi` on PyPI and `testpypi` on TestPyPI.
2. **GitHub environments** (Settings, Environments): create `testpypi` with no
   protection, and `pypi` with *Required reviewers* set to the maintainers.
   Approving that environment is the "release to real PyPI" button. Optionally
   limit both to deployments from `v*` tags.
3. PyPI already has 0.1.2 and the repo has no tags, so the first tag defines
   the next version and must be greater than 0.1.2 (for example `v0.2.0`).

## Cutting a release

1. Make sure `master` is green and `CHANGELOG.md` has an entry under
   **Unreleased** for every user-visible change.
2. Promote the changelog, review it, and merge the commit to `master`:

   ```
   python tools/release_notes.py promote --version 0.2.0
   git commit -am "docs: Prepare 0.2.0 release"
   ```

3. Tag the merge commit and push the tag:

   ```
   git tag -s v0.2.0 -m "v0.2.0"
   git push origin v0.2.0
   ```

4. Watch the **Release** workflow in the Actions tab. It will:
   - verify the tag is on `master` and `CHANGELOG.md` has a `[0.2.0]` section;
   - build the sdist and wheels (Linux x86_64 and arm64, macOS, Windows);
   - publish to TestPyPI and install the result in a clean environment;
   - **wait for approval** of the `pypi` environment.
5. Check the TestPyPI page, then click *Review deployments* and approve. The
   workflow publishes to PyPI and creates the GitHub release from the
   changelog notes.

## Notes

- **Dry run:** run the Release workflow by hand from the Actions tab (any
  branch). It builds every artifact and publishes nothing.
- **Pre-releases:** tag `v0.3.0rc1`. The changelog needs a `[0.3.0rc1]`
  section, so promote with `--version 0.3.0rc1`. For the final release, rename
  that section by hand instead of promoting. GitHub marks the release as a
  pre-release.
- **Failures:** TestPyPI uploads are idempotent (`skip-existing`), so a failed
  run can be re-run. PyPI versions are immutable: if a bad version reached
  PyPI, yank it and release a new version, never reuse a tag.
- **Versioning:** [Semantic Versioning](https://semver.org/). While below 1.0,
  minor releases may contain breaking changes; call them out in the changelog.
