## Contributing to mteb

We welcome contributions to `mteb` such as new tasks, code optimization or benchmarks.

Once you have decided on your contribution, this document describes how to set up the repository for development.


### Development Installation

If you want to submit a task or on other ways contribute to `mteb`, you will need to install the package in development mode:

```bash
# download the git repository
git clone https://github.com/embeddings-benchmark/mteb
cd mteb

# create your virtual environment and activate it
make install
```

This uses [make](https://www.gnu.org/software/make/) to define the install command. You can see what each command does in the [makefile](https://github.com/embeddings-benchmark/mteb/blob/main/Makefile).  

### Running Tests

To run the tests, you can use the following command:

```bash
make test
```

This is also run by the CI pipeline, so if this passed locally, you can be almost sure that your changes will not cause a failed test once you create a pull request. We recommend running the tests in the lowest version of python supported by the package (see the [pyproject.toml](https://github.com/embeddings-benchmark/mteb/blob/main/pyproject.toml)) to ensure compatibility.


### Running linting

To run the linting before submitting a pull request, use:

```bash
make lint
```

This command is equivalent to the command run during CI. It will check for code style and formatting issues.


## Semantic Versioning and Releases

`mteb` follows [semantic versioning](https://semver.org/). This means that the version number of the package is composed of three numbers: `MAJOR.MINOR.PATCH`. This allow us to use existing tools to automatically manage the versioning of the package. For maintainers (and contributors), this means that commits with the following prefixes will automatically trigger a version bump:

- `fix:` for patches
- `feat:` for minor versions
- `breaking:` for major versions

Any commit with one of these prefixes will trigger a version bump upon merging to the main branch as long as tests pass. A version bump will then trigger a new release on PyPI as well as a new release on GitHub.

Other prefixes will not trigger a version bump. For example, `docs:`, `chore:`, `refactor:`, etc., however they will structure the commit history and the changelog. You can find more information about this in the [python-semantic-release documentation](https://python-semantic-release.readthedocs.io/en/latest/). If you do not intend to trigger a version bump you're not required to follow this convention when contributing to `mteb`.