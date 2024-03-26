## Contributing to MTEB
We welcome contributions such as new datasets to MTEB! This section describes how to set up the repository for development.

### Development Installation
If you want to submit a dataset or on other ways contribute to MTEB, you can install the package in development mode:

```bash
git clone https://github.com/embeddings-benchmark/mteb
cd mteb

# create your virtual environment and activate it
make install
```

### Running Tests
To run the tests, you can use the following command:

```bash
make test
# or if you want to run on multiple cores
make test-parallel
```

### Running linting
To run the linting before a PR you can use the following command:

```bash
make lint
```

## Semantic Versioning and Releases
MTEB follows [semantic versioning](https://semver.org/). This means that the version number of the package is composed of three numbers: `MAJOR.MINOR.PATCH`. This allow us to use existing tools to automatically manage the versioning of the package. For maintainers (and contributors), this mean that commits with the following prefixes will automatically trigger a version bump:

- `fix:` for patches
- `feat:` for minor versions
- `breaking:` for major versions

Any commit with one of these prefixes will trigger a version bump upon merging to the main branch as long tests pass. A version bump will then trigger a new release on PyPI as well as a new release on GitHub.

Other prefixes will not trigger a version bump. For example, `docs:`, `chore:`, `refactor:`, etc., however they will structure the commit history and the changelog. You can find more information about this in the [python-semantic-release documentation](https://python-semantic-release.readthedocs.io/en/latest/). If you do not intend to trigger a version bump your are not required to follow this convention when contributing to MTEB.