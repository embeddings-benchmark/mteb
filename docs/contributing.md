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
