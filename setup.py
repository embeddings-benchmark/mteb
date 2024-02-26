"""MTEB is an open library for benchmarking embeddings.
Note:
   VERSION needs to be formatted following the MAJOR.MINOR.PATCH convention
   (we need to follow this convention to be able to retrieve versioned scripts)
Inspired by: https://github.com/huggingface/datasets/blob/main/setup.py
To create the package for pypi.
0. Prerequisites:
   - Dependencies:
     - twine: "pip install twine"
     - wheel: "pip install wheel"
   - Create an account in (and join the 'datasets' project):
     - PyPI: https://pypi.org/
     - Test PyPI: https://test.pypi.org/
1. Change the version in:
   - mteb/__init__.py
   - setup.py
2. Commit these changes: "git commit -m 'Release: VERSION'"
3. Add a tag in git to mark the release: "git tag VERSION -m 'Add tag VERSION for pypi'"
   Push the tag to remote: git push --tags origin main
4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).
   First, delete any "build" directory that may exist from previous builds.
   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).
   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.
5. OPTIONAL: Check that everything looks correct by uploading the package to the pypi test server:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
   Check that you can install it in a virtualenv/notebook by running:
   pip install huggingface_hub fsspec aiohttp
   pip install -U tqdm
   pip install -i https://testpypi.python.org/pypi datasets
6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi
7. Fill release notes in the tag in github once everything is looking hunky-dory.
8. Change the version in __init__.py and setup.py to X.X.X+1.dev0 (e.g. VERSION=1.18.3 -> 1.18.4.dev0).
   Then push the change with a message 'set dev version'
"""


from setuptools import find_packages, setup


with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="mteb",
    version="1.1.3.dev0",
    description="Massive Text Embedding Benchmark",
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords="deep learning, text embeddings, benchmark",
    license="Apache",
    author="MTEB Contributors (https://github.com/embeddings-benchmark/mteb/graphs/contributors)",
    author_email="niklas@huggingface.co, nouamane@huggingface.co, info@nils-reimers.de",
    url="https://github.com/embeddings-benchmark/mteb",
    project_urls={
        "Huggingface Organization": "https://huggingface.co/mteb",
        "Source Code": "https://github.com/embeddings-benchmark/mteb",
    },
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mteb=mteb.cmd:main",
        ]
    },
    python_requires=">=3.7.0",
    install_requires=[
        "datasets>=2.2.0",
        "jsonlines",
        "numpy",
        "requests>=2.26.0",
        "scikit_learn>=1.0.2",
        "scipy",
        "sentence_transformers>=2.2.0",
        "torch",
        "tqdm",
        "rich",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
