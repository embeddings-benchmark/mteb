# CHANGELOG



## v1.3.2 (2024-03-29)

### Documentation

* docs: Update links in README.md (#296) ([`76056b5`](https://github.com/embeddings-benchmark/mteb/commit/76056b5ba92dcbfe32d629897ab6d5db3a0861c4))

### Fix

* fix: Added tasks from SEB (#287)

* Added tasks from SEB

* docs: fix link

* fix: ran linting

* fix typing for 3.8

* fixed annotation for v3.8 ([`39cff49`](https://github.com/embeddings-benchmark/mteb/commit/39cff490157ae87d1cf62c77022f325be729bf04))


## v1.3.1 (2024-03-26)

### Fix

* fix: updated version in transition to semantic release ci ([`238ab82`](https://github.com/embeddings-benchmark/mteb/commit/238ab825e9b221c363589eed89273481e058c50f))


## v1.3.0 (2024-03-26)

### Breaking

* feat: Updating version

BREAKING CHANGE: Bump version ([`caee2e9`](https://github.com/embeddings-benchmark/mteb/commit/caee2e9451999633476fb3305fb3fdc928ec9f0b))

### Ci

* ci: disable changelog ([`b7d3cde`](https://github.com/embeddings-benchmark/mteb/commit/b7d3cde561200264d74e592a678f0dea2eb68129))

* ci: moved release to the correct folder ([`b4fa85a`](https://github.com/embeddings-benchmark/mteb/commit/b4fa85a51374b78b47789557c5467700b859eba5))

* ci: renamed test job and workflow (#282)

ci: Added tests ([`6675bb8`](https://github.com/embeddings-benchmark/mteb/commit/6675bb8668ff17ca8cf3cce2703f3ebf17795bfc))

### Documentation

* docs: typos in readme (#268) ([`aa9234c`](https://github.com/embeddings-benchmark/mteb/commit/aa9234cc24f6dd3408961895d092ee019551fab2))

* docs: add dataset schemas (#255)

* docs: update AbsTaskClassification.py document schema for classification task

* update AbsTaskBitextMining.py

* update BornholmskBitextMining.py

* update AbsTaskClustering.py and BlurbsClusteringP2P.py

* update 8 files

* update 9 files

* update AbsTaskReranking.py

* update BlurbsClusteringP2P.py

* update CMTEBPairClassification.py

* update GerDaLIRRetrieval.py

* update 7 files

* update AbsTaskBitextMining.py

* update AbsTaskClassification.py ([`c3ce1ac`](https://github.com/embeddings-benchmark/mteb/commit/c3ce1ac8ac92baf9a7481c30d476b45e3ec36786))

* docs: Add development installation instructions (#246)

* docs: Add development installation instructions

* removed unused requirements file

I don&#39;t believe this is nec. with the setup.py specifying the same dependencies

* docs: Updated make file with new dependencies

* ci: Update ci to use make commands

This ensure that the user runs exactly what the CI expects

* ci: Avoid specifying tests folder as it causes issuew ith tests

* ci: removed unec. args for test ci

* Added dev install ([`0048878`](https://github.com/embeddings-benchmark/mteb/commit/0048878deba9f57147c3696dcb89ade098c90376))

### Feature

* feat: bump version again ([`294ab91`](https://github.com/embeddings-benchmark/mteb/commit/294ab910f6aa4099c0de8c9f91dbee38efd91aab))

* feat: bump version again ([`acf68c7`](https://github.com/embeddings-benchmark/mteb/commit/acf68c799133d390baba15cdf87f81c844c5a682))

### Fix

* fix: dead link in readme ([`ecbb776`](https://github.com/embeddings-benchmark/mteb/commit/ecbb776fba460c531f09e7b0ce986f075f2b665a))

* fix: Added sizes to the metadata (#276)

* restructing the readme

* added mmteb

* removed unec. method

* Added docstring to metadata

* Updated outdated examples

* formatting documents

* fix: Updated form to be parsed correctly

* fix: Added sizes to the metadata

this allow for automatic metadata generations

* Updated based on feedback

* Apply suggestions from code review

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* updated based on feedback

* Added suggestion from review

* added correction based on review

* reformatted empty fields to None

---------

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`cd4a012`](https://github.com/embeddings-benchmark/mteb/commit/cd4a012271463b89db7a8ec9ca298a975805988d))

### Refactor

* refactor: add metadata basemodel (#260)

* refactor: rename description to metadata dict

* refactor: add TaskMetadata and first example

* update 9 files

* update TaskMetadata.py

* update TaskMetadata.py

* update TaskMetadata.py

* update LICENSE, TaskMetadata.py and requirements.dev.txt

* update 151 files

* update 150 files

* update 43 files and delete 1 file

* update 106 files

* update 45 files

* update 6 files

* update 14 files

* Added model results to repo and updated CLI to create consistent folder structure. (#254)

* Added model results to repo and updated CLI to create consistent folder structure.

* ci: updated ci to use make install

* Added missing pytest dependencies

* Update README.md

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

---------

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Restructing the readme (#262)

* restructing the readme

* removed double specification of versions and moved all setup to pyproject.toml

* correctly use flat-layout for the package

* build(deps): update TaskMetadata.py and pyproject.toml

* update 221 files

* build(deps): update pyproject.toml

* build(deps): update pyproject.toml

* build(deps): update pyproject.toml

---------

Co-authored-by: Kenneth Enevoldsen &lt;kennethcenevoldsen@gmail.com&gt;
Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`dd5d617`](https://github.com/embeddings-benchmark/mteb/commit/dd5d61724e71b2cdba9f9cf7e01fbed1b81cb423))

### Unknown

* overwrite version ([`bc60c9d`](https://github.com/embeddings-benchmark/mteb/commit/bc60c9dc9f9a8a10cae4794a764a01ee766b7659))

* v1.3.0 ([`50b856c`](https://github.com/embeddings-benchmark/mteb/commit/50b856cd116fbcf93b4848ab4c0e58a67d88ca4a))

* v1.3.0 ([`61c12d8`](https://github.com/embeddings-benchmark/mteb/commit/61c12d8cc7f27bea50322cf9332936fa15ccca1a))

* Merge branch &#39;main&#39; of https://github.com/embeddings-benchmark/mteb ([`7b0a766`](https://github.com/embeddings-benchmark/mteb/commit/7b0a76670aa013291bbf52c98b813667d29f3ea1))

* Ci-fix (#289)

* added release pipeline

* v1.3.0

* ci: moved release to the correct folder ([`7f56c1a`](https://github.com/embeddings-benchmark/mteb/commit/7f56c1a7d2eb2fab6eb028291d85054727c650d1))

* Merge branch &#39;main&#39; of https://github.com/embeddings-benchmark/mteb ([`57f500f`](https://github.com/embeddings-benchmark/mteb/commit/57f500f59dcdc27ac7edfed0475b20becc16b191))

* v1.3.0

* added release pipeline

* v1.3.0 ([`5e4d10e`](https://github.com/embeddings-benchmark/mteb/commit/5e4d10e5224d21e51d0ffcd87abee42008f2446c))

* v1.3.0 ([`cdda2f2`](https://github.com/embeddings-benchmark/mteb/commit/cdda2f2786ce582af2a75745873e207739f7f819))

* added release pipeline ([`69a440b`](https://github.com/embeddings-benchmark/mteb/commit/69a440b8a725493f63c63b06f7357648a7e9b37e))

* tests: speed up tests (#283)

update Makefile and test_all_abstasks.py ([`2155bf6`](https://github.com/embeddings-benchmark/mteb/commit/2155bf66c2ea5435744c47b03e3f14b6a5df1813))

* update TaskMetadata.py (#281) ([`acfd7d4`](https://github.com/embeddings-benchmark/mteb/commit/acfd7d420fc3aa05d624db43c5b41f85b1a93367))

* Merge branch &#39;main&#39; of https://github.com/embeddings-benchmark/mteb ([`c9d1a03`](https://github.com/embeddings-benchmark/mteb/commit/c9d1a03c7d8531a0293cbd24e523e904c2be9477))

* Enable ruff ci (#279)

* restructing the readme

* added mmteb

* removed unec. method

* Added docstring to metadata

* Updated outdated examples

* formatting documents

* fix: Updated form to be parsed correctly

* fix: Added sizes to the metadata

this allow for automatic metadata generations

* Updated based on feedback

* Apply suggestions from code review

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* updated based on feedback

* Added suggestion from review

* added correction based on review

* reformatted empty fields to None

* CI: Enable linter

---------

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`a16eb07`](https://github.com/embeddings-benchmark/mteb/commit/a16eb07da1d1a6d8380683e9fa11df3244fae87b))

* Added MMTEB (#275)

* restructing the readme

* added mmteb

* removed unec. method

* Added docstring to metadata

* Updated outdated examples

* formatting documents

* fix: Updated form to be parsed correctly

* Updated based on feedback

* Apply suggestions from code review

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* updated based on feedback

* Added suggestion from review

* added correction based on review

---------

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`c0dc49a`](https://github.com/embeddings-benchmark/mteb/commit/c0dc49a6b99f4d8136b7ec46c49563d7e1b866db))

* dev: add ruff as suggested extension (#274) ([`b08913f`](https://github.com/embeddings-benchmark/mteb/commit/b08913f8616c580f8bbb15bfa808549e2b74912a))

* dev: add isort (#271)

* dev: add isort

* dev: add isort ([`845099d`](https://github.com/embeddings-benchmark/mteb/commit/845099d5b49b0757cc4cf23c08c6d7f65627538e))

* dev: run tests on pull request towards any branch ([`13f759a`](https://github.com/embeddings-benchmark/mteb/commit/13f759a62bff085e156e4d115f64604c2dc0f087))

* Merge branch &#39;main&#39; of https://github.com/embeddings-benchmark/mteb ([`b42abe4`](https://github.com/embeddings-benchmark/mteb/commit/b42abe4e37f96dd0cab898b381d644d507a228a1))

* replaced linter with ruff (#265)

* restructing the readme

* removed double specification of versions and moved all setup to pyproject.toml

* correctly use flat-layout for the package

* replaced linter with ruff

* rerun tests

* ci: Added in newer workflow

some of them are disables as they require other issues to be solved

* Update Makefile

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

---------

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`023e881`](https://github.com/embeddings-benchmark/mteb/commit/023e8817f108a76718fc37f7c8937e000de56786))

* Restructing the readme (#262)

* restructing the readme

* removed double specification of versions and moved all setup to pyproject.toml

* correctly use flat-layout for the package ([`769157b`](https://github.com/embeddings-benchmark/mteb/commit/769157b1e49c97ee6ca334a299392392bc3a6523))

* restructing the readme ([`364be7f`](https://github.com/embeddings-benchmark/mteb/commit/364be7f4a26275263c9a82de594e47aaf28a1bcf))

* Added model results to repo and updated CLI to create consistent folder structure. (#254)

* Added model results to repo and updated CLI to create consistent folder structure.

* ci: updated ci to use make install

* Added missing pytest dependencies

* Update README.md

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

---------

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`8a758bc`](https://github.com/embeddings-benchmark/mteb/commit/8a758bce00a6bc64dd4f0dab98f5bc3e0c683f46))

* dev: add workspace defaults in VSCode (#253)

* dev: add black as default formatter in vscode

* Update .vscode/settings.json

---------

Co-authored-by: Kenneth Enevoldsen &lt;kennethcenevoldsen@gmail.com&gt; ([`30e5b9e`](https://github.com/embeddings-benchmark/mteb/commit/30e5b9ebb288d923b3c7e8d0e55728f82a1bc5d8))

* Add Danish Discourse dataset (#247)

* misc.

* update ddisco.py

* chore: delete ddisco.py, ddisco.test.tsv and ddisco.train.tsv

* Update mteb/tasks/Classification/DdiscoCohesionClassification.py

Co-authored-by: Kenneth Enevoldsen &lt;kennethcenevoldsen@gmail.com&gt;

* Update mteb/tasks/Classification/DdiscoCohesionClassification.py

Co-authored-by: Kenneth Enevoldsen &lt;kennethcenevoldsen@gmail.com&gt;

* Update mteb/tasks/Classification/DdiscoCohesionClassification.py

Co-authored-by: Imene Kerboua &lt;33312980+imenelydiaker@users.noreply.github.com&gt;

* Update mteb/tasks/Classification/DdiscoCohesionClassification.py

Co-authored-by: Imene Kerboua &lt;33312980+imenelydiaker@users.noreply.github.com&gt;

* Update mteb/tasks/Classification/DdiscoCohesionClassification.py

Co-authored-by: Imene Kerboua &lt;33312980+imenelydiaker@users.noreply.github.com&gt;

---------

Co-authored-by: Kenneth Enevoldsen &lt;kennethcenevoldsen@gmail.com&gt;
Co-authored-by: Imene Kerboua &lt;33312980+imenelydiaker@users.noreply.github.com&gt; ([`d46d0f5`](https://github.com/embeddings-benchmark/mteb/commit/d46d0f5281d5a9036501aca437e9c76480dc8885))

* Update structure of mteb/tasks to mteb/tasks/{type}/{language}  (#245)

* Fix structure of mteb/tasks
Fixes #243

* fix: Added missing init files ([`b1c78c1`](https://github.com/embeddings-benchmark/mteb/commit/b1c78c1121fbd488d62d1385d6f000d3f3b46ef4))

* tests: do not run tests on collection (#249)

test: update test_all_abstasks.py ([`236614a`](https://github.com/embeddings-benchmark/mteb/commit/236614add5de4e848bc0f1db8ad997f451ae2906))

* Update README.md with correct DRESModel location ([`399edf4`](https://github.com/embeddings-benchmark/mteb/commit/399edf4331cd316b28a832fd37e187d5c5e204f1))

* Fix typo ([`9610378`](https://github.com/embeddings-benchmark/mteb/commit/96103788cd135caecdb439b5d26af38ab6cd33ef))

* Set dev version ([`716f59c`](https://github.com/embeddings-benchmark/mteb/commit/716f59cae9be31a747371497ec6792b23070270c))


## v1.2.0 (2024-03-07)

### Unknown

* Release: 1.2.0 ([`9e9dca8`](https://github.com/embeddings-benchmark/mteb/commit/9e9dca890bcced66494996863a2fb52ea4129d87))

* Rmv superfluous file ([`d772fed`](https://github.com/embeddings-benchmark/mteb/commit/d772fedb2c102112217e494987eaf2468925256e))

* Remove duplicate &amp; outdated  code ([`12bcb83`](https://github.com/embeddings-benchmark/mteb/commit/12bcb83835943011c92efaf117f16c808f3a1fe8))

* Adapt scripts ([`36b9234`](https://github.com/embeddings-benchmark/mteb/commit/36b92341e0178a7bc2276cac468f2e73b5c5880c))

* Add example ([`273ff4a`](https://github.com/embeddings-benchmark/mteb/commit/273ff4acb70b063bbcae04d5812f580e1dea4bc2))

* Simplify retrieval (#233)

* Simplify retrieval

* Simplify

* Make __call__ method

* Add splits

* Rmv outdated test

* Fix name &amp; \n

* Add qrels

* Add revisions

Co-authored-by: Imene Kerboua &lt;33312980+imenelydiaker@users.noreply.github.com&gt;

* Add hf hub org

* Add test

* Add missing revision

* Rename test

Co-authored-by: Imene Kerboua &lt;33312980+imenelydiaker@users.noreply.github.com&gt;

* log dres compat

---------

Co-authored-by: Imene Kerboua &lt;33312980+imenelydiaker@users.noreply.github.com&gt; ([`c9fccbc`](https://github.com/embeddings-benchmark/mteb/commit/c9fccbcb7f576e33b6b6f487c8f3dc5cbecc0a33))

* Fixed missing revision error on Norwegian Bitext Mining (#221)

* Removed revision specification from Norwegian Bitext Mining task

* Update to latest revision

---------

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`b249c67`](https://github.com/embeddings-benchmark/mteb/commit/b249c6766622a6c2a2f9d9194e140eca9cbd0e3c))

* Remove HAGRID from french benchmark (#235)

* add Masakhane dataset config

* add trigram lang code for dataset who use it

* create french script eval

* fix French word

* add some documentation

* add script to process and upload alloprof on HF

* build script for HF

* adding dataset processing for mteb

* add script to process and upload alloprof on HF

* build script for HF

* adding dataset processing for mteb

* refactor few thing

* remove whitespaces

* 4 pair classification (#10)

* add Opusparcus dataset

* multilingual usage

* use eval_split of config files

* change eval_split according to data

---------

Co-authored-by: Gabriel Sequeira &lt;gsequeira@openstudio.fr&gt;

* add script to process and upload alloprof on HF

* build script for HF

* adding dataset processing for mteb

* refactor few thing

* remove whitespaces

* Clustering with HAL S2S dataset (#11)

HAL S2S dataset creation and evaluation on clustering task.

* adding BSARD dataset

* add BSARD to benchmark

* adding Hagrid dataset

* DiaBLa and Flores Bitext Mining evaluation (#12)

* Add DiaBLa dataset for bitext mining

* Add DiaBLa dataset for bitext mining

* deduplicate bitext task

* add Flores

* format files

* add flores to evaluation script

* remove prints

* add revision

---------

Co-authored-by: Gabriel Sequeira &lt;gsequeira@openstudio.fr&gt;

* add script to process and upload alloprof on HF

* build script for HF

* adding dataset processing for mteb

* refactor few thing

* remove whitespaces

* adding dataset processing for mteb

* adding BSARD dataset

* add BSARD to benchmark

* adding Hagrid dataset

* fix change on langmapping

* reset alphabetical order

* add revision handling

* Clustering: Add AlloProf dataset  (#17)

AlloProf dataset for clustering task

* handling of revision

* change split + add revision handling

* add script to process and upload alloprof on HF

* build script for HF

* adding dataset processing for mteb

* refactor few thing

* remove whitespaces

* adding dataset processing for mteb

* adding BSARD dataset

* add BSARD to benchmark

* adding Hagrid dataset

* add script to process and upload alloprof on HF

* adding dataset processing for mteb

* refactor few thing

* reset alphabetical order

* add revision handling

* handling of revision

* change split + add revision handling

* use eval variable

* alphabetic order

* Add MLSUM dataset for clustering task (#21)

* Use Masakhane dataset for clustering task (#23)

* 16 add datasets to readmemd (#18)

* run task table

* run task table

* Add MLSUM dataset for clustering task (#21)

* Use Masakhane dataset for clustering task (#23)

* run task table

* refresh readme

* refresh readme

* run task table

* refresh readme

---------

Co-authored-by: Gabriel Sequeira &lt;gsequeira@openstudio.fr&gt;
Co-authored-by: Marion Schaeffer &lt;92590517+schmarion@users.noreply.github.com&gt;

* load only test split (#25)

Co-authored-by: Gabriel Sequeira &lt;gsequeira@openstudio.fr&gt;

* Update mteb/tasks/BitextMining/DiaBLaBitextMining.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/tasks/Clustering/HALClusteringS2S.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* renaming masakhane (#28)

Co-authored-by: Gabriel Sequeira &lt;gsequeira@openstudio.fr&gt;

* Syntec dataset addition (#26)

* add scrpit to process &amp; load to HF

* add script to enable download of data from HF

* add syntec dataset files to gitignore

* add syntecretrieval

* add syntec retrival

* build dataloading script

* remove datasets

* correct typo

---------

Co-authored-by: Sequeira Gabriel &lt;gabriel.sequeira@outlook.fr&gt;

* 30 add syntec reranking (#31)

* change name to secify retrieval

* add reranking tasks

* create script to upload dataset fo reranking task

* create reranking task

* add reranking tasks

* add model name in description

* SummEval translated to french (#32)

* 7 sts (#33)

* taike into account multilingual tasks

* add stsbenchmark multilingual dataset

* add STS tasks

* taike into account multilingual tasks

* add stsbenchmark multilingual dataset

* add STS tasks

* add coma

* Adding sick fr dataset to sts tasks (#34)

* Adding sick fr dataset to sts tasks
* modifying dataset in load function to have the right column names

* Fix alloprof dataset (#36)

* change revision to use

* remove duplicate data

* change main metric because dataset is hard (#37)

* Fix alloprof dataset (#40)

* change revision to use

* remove duplicate data

* change revision

* handle queries train test split

* change dataset creation method

* change revision

* handle queries train test split

* change dataset creation method

* Fix DiaBLa by inheriting CrossLingual class (#42)

* Fix DiaBLa by inheriting CrossLingual class

* remove remaining print

* Fix DiaBLa integration

* Update mteb/tasks/BitextMining/FloresBitextMining.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update README.md

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update README.md

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/tasks/Classification/MasakhaNEWSClassification.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update README.md

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update README.md

* Update mteb/tasks/BitextMining/FloresBitextMining.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/evaluation/MTEB.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/abstasks/AbsTaskPairClassification.py

Co-authored-by: Imene Kerboua &lt;33312980+imenelydiaker@users.noreply.github.com&gt;

* Update README.md

* Update scripts/data/syntec/create_data_reranking.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update scripts/data/alloprof/create_data_reranking.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update scripts/run_mteb_french.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update scripts/run_mteb_french.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/evaluation/MTEB.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/evaluation/MTEB.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/tasks/Retrieval/HagridRetrieval.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/tasks/Clustering/MLSUMClusteringP2P.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/tasks/Clustering/MLSUMClusteringS2S.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/tasks/Clustering/MasakhaNEWSClusteringP2P.py

* Update mteb/tasks/Clustering/MasakhaNEWSClusteringS2S.py

* Update mteb/tasks/STS/SickFrSTS.py

* Inherit OpusparcusPC init from MultilingualTask

* remove unnecessary init

* Remove train split from evaluation on MasakhaNEWSClassification (#52)

remove train split from evaluation

* put script on HF dataset repos (#56)

* put script on HF dataset repos

* remove scripts

* 49 fix dictionnary in syntecretrieval (#54)

* add trust remote code arg

* leave corpus as dict

* remove trust remote code

* add Tatoeba &amp; BUCC BitextMining tasks (#57)

add bucc and tatoeba bitextmining tasks

* 46 add other languages to masakhaneweclusterings2s and p2p (#58)

* add other language to clustering tasks

* fix main score and S2S task

* update run fr becnhmark script

* Update run_mteb_french.py

* Update AbsTaskClustering.py

* remove train and validation splits

* remove Hagrid (#60)

---------

Co-authored-by: Gabriel Sequeira &lt;gsequeira@openstudio.fr&gt;
Co-authored-by: Marion Schaeffer &lt;92590517+schmarion@users.noreply.github.com&gt;
Co-authored-by: mciancone@openstudio.fr &lt;mciancone@openstudio.fr&gt;
Co-authored-by: Sequeira Gabriel &lt;gabriel.sequeira@outlook.fr&gt;
Co-authored-by: Imene Kerboua &lt;33312980+imenelydiaker@users.noreply.github.com&gt;
Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;
Co-authored-by: wissam-sib &lt;36303760+wissam-sib@users.noreply.github.com&gt;
Co-authored-by: Wissam Siblini &lt;wissam.siblini92@gmail.com&gt; ([`d01d053`](https://github.com/embeddings-benchmark/mteb/commit/d01d053028362dc1568be6b8fcff8be915d97837))

* Restore TRECCOVID import ([`9f8e897`](https://github.com/embeddings-benchmark/mteb/commit/9f8e897b1edf80e683dcfa15931d8070d496ffd6))

* Extend MTEB with French datasets (#218)

* add Masakhane dataset config

* add trigram lang code for dataset who use it

* create french script eval

* fix French word

* add some documentation

* add script to process and upload alloprof on HF

* build script for HF

* adding dataset processing for mteb

* add script to process and upload alloprof on HF

* build script for HF

* adding dataset processing for mteb

* refactor few thing

* remove whitespaces

* 4 pair classification (#10)

* add Opusparcus dataset

* multilingual usage

* use eval_split of config files

* change eval_split according to data

---------

Co-authored-by: Gabriel Sequeira &lt;gsequeira@openstudio.fr&gt;

* add script to process and upload alloprof on HF

* build script for HF

* adding dataset processing for mteb

* refactor few thing

* remove whitespaces

* Clustering with HAL S2S dataset (#11)

HAL S2S dataset creation and evaluation on clustering task.

* adding BSARD dataset

* add BSARD to benchmark

* adding Hagrid dataset

* DiaBLa and Flores Bitext Mining evaluation (#12)

* Add DiaBLa dataset for bitext mining

* Add DiaBLa dataset for bitext mining

* deduplicate bitext task

* add Flores

* format files

* add flores to evaluation script

* remove prints

* add revision

---------

Co-authored-by: Gabriel Sequeira &lt;gsequeira@openstudio.fr&gt;

* add script to process and upload alloprof on HF

* build script for HF

* adding dataset processing for mteb

* refactor few thing

* remove whitespaces

* adding dataset processing for mteb

* adding BSARD dataset

* add BSARD to benchmark

* adding Hagrid dataset

* fix change on langmapping

* reset alphabetical order

* add revision handling

* Clustering: Add AlloProf dataset  (#17)

AlloProf dataset for clustering task

* handling of revision

* change split + add revision handling

* add script to process and upload alloprof on HF

* build script for HF

* adding dataset processing for mteb

* refactor few thing

* remove whitespaces

* adding dataset processing for mteb

* adding BSARD dataset

* add BSARD to benchmark

* adding Hagrid dataset

* add script to process and upload alloprof on HF

* adding dataset processing for mteb

* refactor few thing

* reset alphabetical order

* add revision handling

* handling of revision

* change split + add revision handling

* use eval variable

* alphabetic order

* Add MLSUM dataset for clustering task (#21)

* Use Masakhane dataset for clustering task (#23)

* 16 add datasets to readmemd (#18)

* run task table

* run task table

* Add MLSUM dataset for clustering task (#21)

* Use Masakhane dataset for clustering task (#23)

* run task table

* refresh readme

* refresh readme

* run task table

* refresh readme

---------

Co-authored-by: Gabriel Sequeira &lt;gsequeira@openstudio.fr&gt;
Co-authored-by: Marion Schaeffer &lt;92590517+schmarion@users.noreply.github.com&gt;

* load only test split (#25)

Co-authored-by: Gabriel Sequeira &lt;gsequeira@openstudio.fr&gt;

* Update mteb/tasks/BitextMining/DiaBLaBitextMining.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/tasks/Clustering/HALClusteringS2S.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* renaming masakhane (#28)

Co-authored-by: Gabriel Sequeira &lt;gsequeira@openstudio.fr&gt;

* Syntec dataset addition (#26)

* add scrpit to process &amp; load to HF

* add script to enable download of data from HF

* add syntec dataset files to gitignore

* add syntecretrieval

* add syntec retrival

* build dataloading script

* remove datasets

* correct typo

---------

Co-authored-by: Sequeira Gabriel &lt;gabriel.sequeira@outlook.fr&gt;

* 30 add syntec reranking (#31)

* change name to secify retrieval

* add reranking tasks

* create script to upload dataset fo reranking task

* create reranking task

* add reranking tasks

* add model name in description

* SummEval translated to french (#32)

* 7 sts (#33)

* taike into account multilingual tasks

* add stsbenchmark multilingual dataset

* add STS tasks

* taike into account multilingual tasks

* add stsbenchmark multilingual dataset

* add STS tasks

* add coma

* Adding sick fr dataset to sts tasks (#34)

* Adding sick fr dataset to sts tasks
* modifying dataset in load function to have the right column names

* Fix alloprof dataset (#36)

* change revision to use

* remove duplicate data

* change main metric because dataset is hard (#37)

* Fix alloprof dataset (#40)

* change revision to use

* remove duplicate data

* change revision

* handle queries train test split

* change dataset creation method

* change revision

* handle queries train test split

* change dataset creation method

* Fix DiaBLa by inheriting CrossLingual class (#42)

* Fix DiaBLa by inheriting CrossLingual class

* remove remaining print

* Fix DiaBLa integration

* Update mteb/tasks/BitextMining/FloresBitextMining.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update README.md

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update README.md

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/tasks/Classification/MasakhaNEWSClassification.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update README.md

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update README.md

* Update mteb/tasks/BitextMining/FloresBitextMining.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/evaluation/MTEB.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/abstasks/AbsTaskPairClassification.py

Co-authored-by: Imene Kerboua &lt;33312980+imenelydiaker@users.noreply.github.com&gt;

* Update README.md

* Update scripts/data/syntec/create_data_reranking.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update scripts/data/alloprof/create_data_reranking.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update scripts/run_mteb_french.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update scripts/run_mteb_french.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/evaluation/MTEB.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/evaluation/MTEB.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/tasks/Retrieval/HagridRetrieval.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/tasks/Clustering/MLSUMClusteringP2P.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/tasks/Clustering/MLSUMClusteringS2S.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/tasks/Clustering/MasakhaNEWSClusteringP2P.py

* Update mteb/tasks/Clustering/MasakhaNEWSClusteringS2S.py

* Update mteb/tasks/STS/SickFrSTS.py

* Inherit OpusparcusPC init from MultilingualTask

* remove unnecessary init

* Remove train split from evaluation on MasakhaNEWSClassification (#52)

remove train split from evaluation

* put script on HF dataset repos (#56)

* put script on HF dataset repos

* remove scripts

* 49 fix dictionnary in syntecretrieval (#54)

* add trust remote code arg

* leave corpus as dict

* remove trust remote code

* add Tatoeba &amp; BUCC BitextMining tasks (#57)

add bucc and tatoeba bitextmining tasks

* 46 add other languages to masakhaneweclusterings2s and p2p (#58)

* add other language to clustering tasks

* fix main score and S2S task

* update run fr becnhmark script

* Update run_mteb_french.py

* Update AbsTaskClustering.py

* remove train and validation splits

---------

Co-authored-by: Gabriel Sequeira &lt;gsequeira@openstudio.fr&gt;
Co-authored-by: Marion Schaeffer &lt;92590517+schmarion@users.noreply.github.com&gt;
Co-authored-by: mciancone@openstudio.fr &lt;mciancone@openstudio.fr&gt;
Co-authored-by: Imene Kerboua &lt;33312980+imenelydiaker@users.noreply.github.com&gt;
Co-authored-by: mciancone &lt;73994289+Sunalwing@users.noreply.github.com&gt;
Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;
Co-authored-by: wissam-sib &lt;36303760+wissam-sib@users.noreply.github.com&gt;
Co-authored-by: Wissam Siblini &lt;wissam.siblini92@gmail.com&gt; ([`3d8b8ec`](https://github.com/embeddings-benchmark/mteb/commit/3d8b8ec563963905923fd1627093c7779d77a5cd))

* dev ([`c16eddc`](https://github.com/embeddings-benchmark/mteb/commit/c16eddcd111dd25e5a3526bcce9d7ba162a28fd5))

* Dev ([`08c7317`](https://github.com/embeddings-benchmark/mteb/commit/08c7317ba76dea38f0afb51278391e20ab2463e0))

* Add tasks for Spanish Embedding Evaluation (#227)

* feat: add xmarket es dataset

* refactor: use multilingual dataset

* fix: update revision id

* refactor: add constant for language

* feat: add two clustering datasets

Signed-off-by: jupyterjazz &lt;saba.sturua@jina.ai&gt;

* feat: import classes

Signed-off-by: jupyterjazz &lt;saba.sturua@jina.ai&gt;

* refactor: flores dataset

Signed-off-by: jupyterjazz &lt;saba.sturua@jina.ai&gt;

* feat: add miracl reranking task for spanish

* feat: use hf repo with all reranking langs

* feat: update revision hash

* refactor: use description for language

* feat: add stses task

* fix: get scores from label column

* refactor: add revision to data loading

* Added spanish passage retrieval

* feat: mintaka and xpqa retrieval tasks

Signed-off-by: jupyterjazz &lt;saba.sturua@jina.ai&gt;

* feat: import classes

Signed-off-by: jupyterjazz &lt;saba.sturua@jina.ai&gt;

* fix: typo in data loading

* fix: id

Signed-off-by: jupyterjazz &lt;saba.sturua@jina.ai&gt;

* refactor: try out multilingual task

Signed-off-by: jupyterjazz &lt;saba.sturua@jina.ai&gt;

* refactor: multilingual task import

Signed-off-by: jupyterjazz &lt;saba.sturua@jina.ai&gt;

* refactor: cmon man

Signed-off-by: jupyterjazz &lt;saba.sturua@jina.ai&gt;

* refactor: go back to monolingual tasks

Signed-off-by: jupyterjazz &lt;saba.sturua@jina.ai&gt;

* refactor: remove unused import

Signed-off-by: jupyterjazz &lt;saba.sturua@jina.ai&gt;

* refactor: loading logic

Signed-off-by: jupyterjazz &lt;saba.sturua@jina.ai&gt;

* feat: add miracl as retrieval task

* fix: nested corpus

* refactor: get lang from description

* Update mteb/tasks/Retrieval/MIRACLRetrieval.py

Co-authored-by: Michael Günther &lt;michael.guenther@jina.ai&gt;

* feat: allow multlingual reranking tasks

* feat: make miraclreranking multilingual

* refactor: rename miraclretrieval

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* style: add missing eof empty line

* feat: make xmarket retrieval task multilingual

* refactor: rename xmarket

* refactor: turn spanish tasks multilingual (#11)

* refactor: make xpqa retrieval multilingual

* fix: formatting of xpqa dataset

* refactor: make mintaka into multilingual task

* refactor: make miracl retrieval multilingual

* feat: add revision ids for hf datasets

* refactor: remove patool

* Update mteb/tasks/Reranking/__init__.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update mteb/tasks/STS/__init__.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

---------

Signed-off-by: jupyterjazz &lt;saba.sturua@jina.ai&gt;
Co-authored-by: guenthermi &lt;guenthermi50@gmail.com&gt;
Co-authored-by: jupyterjazz &lt;saba.sturua@jina.ai&gt;
Co-authored-by: Markus Krimmel &lt;markus.krimmel@jina.ai&gt;
Co-authored-by: Michael Günther &lt;michael.guenther@jina.ai&gt;
Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`52d5c9f`](https://github.com/embeddings-benchmark/mteb/commit/52d5c9f666f8b5f5e65cd9bf3f4651078ff1fced))


## v1.1.2 (2024-02-16)

### Feature

* feat: update revision id of wikicitiesclustering task ([`fb90c02`](https://github.com/embeddings-benchmark/mteb/commit/fb90c022e11834ae6605f5bbb0a79af701793a96))

### Fix

* fix: remove debugging print statement ([`d292d93`](https://github.com/embeddings-benchmark/mteb/commit/d292d937ceedb5d137537b2c25a0f135d1bb91b9))

* fix: pass parallel_retrieval kwarg to use DenseRetrievalParallelExactSearch ([`19b8f66`](https://github.com/embeddings-benchmark/mteb/commit/19b8f6619f07dfd95860f43f9376af230978f447))

### Unknown

* Release: 1.1.2 ([`def3c91`](https://github.com/embeddings-benchmark/mteb/commit/def3c9146ff437d4b9bb690dea183070ccb36a7f))

* Add task list (#228)

* Add task list

* Update mteb/__init__.py

* Update README.md ([`10bf6f8`](https://github.com/embeddings-benchmark/mteb/commit/10bf6f84840ee38ed632861cda603976f87612ec))

* Update BeIRPLTask.py (#225)

* Update BeIRPLTask.py

* Update BeIRPLTask.py ([`a8922c1`](https://github.com/embeddings-benchmark/mteb/commit/a8922c14845c2fa2a20d591f93ffa0cfb42baccc))

* Allow multiple languages ([`2cc222e`](https://github.com/embeddings-benchmark/mteb/commit/2cc222efd84c29cbb5ff04fb8e6703674d53dda9))

* Add Korean Text Search Tasks to MTEB (#210)

* add Ko-miracl, Ko-StrategyQA, Ko-mrtydi tasks

* Update mteb/abstasks/AbsTaskRetrieval.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update AbsTaskRetrieval.py

* Update mteb/abstasks/AbsTaskRetrieval.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* Update scripts/run_mteb_korean.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

---------

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`dadf2da`](https://github.com/embeddings-benchmark/mteb/commit/dadf2dacc1bd12e08d7507597497d6727f4c9720))

* Add MultiLongDocRetrieval task to MTEB. (#224)

* Update AbsTaskRetrieval.py.

* Add Retrieval Task: `MultiLongDocRetrieval`

* Update AbsTaskRetrieval.py and `MLDR` task

* Update reference of MLDR ([`2f65179`](https://github.com/embeddings-benchmark/mteb/commit/2f65179e3ad31b5c46115c620815cc162b23890b))

* Fix name ([`2989f76`](https://github.com/embeddings-benchmark/mteb/commit/2989f76dfd47d719a338ea564ff1122c80b4b51f))

* only save top-k (#209)

* Update AbsTaskRetrieval.py

* Add json import; rename kwarg

* Pass OF

* Update mteb/abstasks/AbsTaskRetrieval.py

* Update AbsTaskRetrieval.py

* Update AbsTaskRetrieval.py

* Update mteb/abstasks/AbsTaskRetrieval.py

---------

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`f58888d`](https://github.com/embeddings-benchmark/mteb/commit/f58888d30439bc8e820b3f894ec12bce5393ba76))

* Add tasks for German Embedding Evaluation (#214)

* chore: solve merge conflict

* fix: gerdalir dataset

* fix: lang from en to de

* chore: solve merge conflict

* chore: add ir datasets to requirements

* refactor: limit queries to 10k

* refactor: update description of task with limit

* revert style changes

* feat: add german stsbenchmarksts task

* feat: update revision id

* refactor: update revision id after changes in scores

* add XMarket dataset

* add xmarket to init file

* feat: add revision id

* add paws x dataset

* Add ir_datasets as dependency

* add GermanDPR dataset

* fix loading

* Update mteb/tasks/Retrieval/GermanDPRRetrieval.py

Co-authored-by: Saba Sturua &lt;45267439+jupyterjazz@users.noreply.github.com&gt;

* feat: add miracl reranking task for german

* refactor: cleanup task

* prevent duplicate pos docs

* fix: use test split in MIRACL (#13)

Fixes mismatch between description and HuggingFace dataset

* refactor: remove WikiCLIR

* fix: double import; xmarket name

* add German tasks to run_mteb_german script

* fupdate revisions and style

* update MIRACL to work with latest version

* revert adding ir_datasets

* support multilingual pair classification

* remove print statement

* Apply suggestions from code review

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt;

* fix monolingual pair classification

* remove lang for monolingual tasks

---------

Co-authored-by: Isabelle Mohr &lt;isabelle.mohr@jina.ai&gt;
Co-authored-by: Markus Krimmel &lt;markus.krimmel@jina.ai&gt;
Co-authored-by: Saba Sturua &lt;45267439+jupyterjazz@users.noreply.github.com&gt;
Co-authored-by: Markus Krimmel &lt;montcyril@gmail.com&gt;
Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`9aba9ee`](https://github.com/embeddings-benchmark/mteb/commit/9aba9ee95a49a48d23b4ff6cfb9cc84bc9dfca32))

* Simplify ([`1cd07db`](https://github.com/embeddings-benchmark/mteb/commit/1cd07db380d7ce51f19653b4f63e18585fcf6398))

* Refer to other works ([`8f28bcb`](https://github.com/embeddings-benchmark/mteb/commit/8f28bcb554d8c605aa9b2c214b3d2b53c070c066))

* Update mteb/tasks/Retrieval/GermanQuADRetrieval.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`09a9cb0`](https://github.com/embeddings-benchmark/mteb/commit/09a9cb0d589214f2d41b0755a75c2ee6609b581a))

* clean up ([`51c40fd`](https://github.com/embeddings-benchmark/mteb/commit/51c40fdd1d913479ce94a0d86b6cd949eda876ab))

* WIP: implement requested changes ([`58baad2`](https://github.com/embeddings-benchmark/mteb/commit/58baad2da110e681a78962eaf794446ad730f960))

* remove code for writing JSONL dataset ([`d23eac3`](https://github.com/embeddings-benchmark/mteb/commit/d23eac312e8e8c067c5f7fb1354829cafee0de3b))

* add docstring, remove local qrels ([`af7ee50`](https://github.com/embeddings-benchmark/mteb/commit/af7ee501008312e9c11550640fcabd529d9ad836))

* fix query id in qrel dataset, ready to merge ([`33c9dd4`](https://github.com/embeddings-benchmark/mteb/commit/33c9dd45aad83c786e7a716fd61cd46471b8140e))

* WIP: use HF dataset instead of local JSONL ([`db3fea1`](https://github.com/embeddings-benchmark/mteb/commit/db3fea1a3803da6af8df5d9f8466d83e436cee83))

* rename BeIRDETask ([`e56cf86`](https://github.com/embeddings-benchmark/mteb/commit/e56cf86c4f6b3a4ded5547d99082303a92bcbe51))

* Update scripts/run_mteb_german.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`4b18a7e`](https://github.com/embeddings-benchmark/mteb/commit/4b18a7e49b07b94f4eff215a1d0f1cc7b69302fd))

* Update mteb/tasks/Retrieval/GermanRetrieval.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`3fef61a`](https://github.com/embeddings-benchmark/mteb/commit/3fef61a3139bd05594f1c3b50fb450f8190f6e3d))

* add reference to GermanQuAD ([`ae268e0`](https://github.com/embeddings-benchmark/mteb/commit/ae268e031ac04ec715944a174de71f1b117ca44f))

* fix results folder path ([`dc7fc01`](https://github.com/embeddings-benchmark/mteb/commit/dc7fc01baabce94ce95d74d08f7782a43ac1dbaf))

* copy from local ([`9c0880d`](https://github.com/embeddings-benchmark/mteb/commit/9c0880d65402bad4862db1b7a4dd27f7379e2863))

* Update mteb/abstasks/AbsTaskRetrieval.py ([`be1fcc1`](https://github.com/embeddings-benchmark/mteb/commit/be1fcc1d47d301c1e627e587430fafd18a7dc316))

* Pass OF ([`b0e6316`](https://github.com/embeddings-benchmark/mteb/commit/b0e63164d8602bb4bb6919d4c6b457ec2726a145))

* Add json import; rename kwarg ([`d39c21c`](https://github.com/embeddings-benchmark/mteb/commit/d39c21c8cdf2b8a027ec67b0c496918835442539))

* Update AbsTaskRetrieval.py ([`4eb8e02`](https://github.com/embeddings-benchmark/mteb/commit/4eb8e02adb57c5f074d34a791703d2e996c283c8))

* Added Norwegian Bokmål-Nynorsk bitext mining task ([`c3fb742`](https://github.com/embeddings-benchmark/mteb/commit/c3fb742eca4143195c70861861288b16ac32fc9c))

* Add STS revisions ([`38277ae`](https://github.com/embeddings-benchmark/mteb/commit/38277ae6f76fe2623373a4e040cc6ddaa592dae7))

* Add RTR revisions ([`8da9487`](https://github.com/embeddings-benchmark/mteb/commit/8da9487f7cea6a3e34e575eb1479be9fe3007c3a))

* Add RRK revisions ([`2011cd8`](https://github.com/embeddings-benchmark/mteb/commit/2011cd8dc040cb38baa48307985629afd91a8b3e))

* Add PCLF revisions ([`9b6f4b9`](https://github.com/embeddings-benchmark/mteb/commit/9b6f4b9ff89103ce621b75afcfd93b49f1f0c222))

* Add CLST revisions ([`da73236`](https://github.com/embeddings-benchmark/mteb/commit/da73236d4a7c1e980895e71033194fe2435b1184))

* Add CLF revisions ([`fd91a9c`](https://github.com/embeddings-benchmark/mteb/commit/fd91a9c439b48eacd0107e62d6545f89d6608b3b))

* Update Revision ([`6b0fae5`](https://github.com/embeddings-benchmark/mteb/commit/6b0fae5799112717e45ee188a3a83b57aee48b2f))

* Fix SweFAQ linkage ([`2341c48`](https://github.com/embeddings-benchmark/mteb/commit/2341c4892a0ef7b8baf5c9aa5906303d0b7e6839))

* Fix SummEval linkage ([`7252322`](https://github.com/embeddings-benchmark/mteb/commit/7252322d0a1b6069e5cbfd827bc71093b0e4bb90))

* Fix Dalaj linkage ([`fb9ccd8`](https://github.com/embeddings-benchmark/mteb/commit/fb9ccd8dde24817455ae3465077436a13b985e7d))

* Fix medrxiv mislinkage ([`620defc`](https://github.com/embeddings-benchmark/mteb/commit/620defcc7b1048eb2cfe4041957196e22686adce))

* Fix stripping ([`02e84b2`](https://github.com/embeddings-benchmark/mteb/commit/02e84b2fa8d147a86b4896d8e57e83f36285f5c7))

* add datasets for long document evaluation

---------

Co-authored-by: Isabelle Mohr &lt;retrospect@protonmail.com&gt; ([`88beb46`](https://github.com/embeddings-benchmark/mteb/commit/88beb46d1340814875fafc27d337e0ed53125a4a))

* Do not enforce rich import ([`aa11fe7`](https://github.com/embeddings-benchmark/mteb/commit/aa11fe723e9b680f07fb01d1365d4975c5fa5803))

* fix RerankingEvaluator&#39;s compute_metrics_individual ([`fd7bfac`](https://github.com/embeddings-benchmark/mteb/commit/fd7bfac8add2a660f651fc02c9681dc4ad6e484a))

* Fix SummEval import ([`859d38e`](https://github.com/embeddings-benchmark/mteb/commit/859d38ec34b91a68f371ea81aade0f23c2d84aec))

* Increment version ([`4d75ddf`](https://github.com/embeddings-benchmark/mteb/commit/4d75ddf448c93b4b879e60e110061f7dcf76ae42))


## v1.1.1 (2023-09-20)

### Fix

* fix: msmarco-v2 uses dev.tsv, not dev1.tsv ([`6908d21`](https://github.com/embeddings-benchmark/mteb/commit/6908d21cfce644140bd70df47df0452c551ee0d0))

* fix: add missing task-langs attribute (#152) ([`bc22909`](https://github.com/embeddings-benchmark/mteb/commit/bc22909c49284efb0df1d997ac23806694424a94))

### Unknown

* Release: 1.1.1 ([`d3aaf4f`](https://github.com/embeddings-benchmark/mteb/commit/d3aaf4f1c1c503535d4d4de0a09e1ab7159dcd93))

* Merge branch &#39;main&#39; into fixconversion ([`d292258`](https://github.com/embeddings-benchmark/mteb/commit/d29225883c1e57f5dd56565080d497905ef9d92a))

* Fix eval_lang ([`7836148`](https://github.com/embeddings-benchmark/mteb/commit/7836148eeab16ad85bd1aaa1bab1d9590b831dbe))

* Simplify code snippets ([`d434f52`](https://github.com/embeddings-benchmark/mteb/commit/d434f5269b96d278b9b1bd286ddc70e5fb76b661))

* Simplify wording ([`3adb0b5`](https://github.com/embeddings-benchmark/mteb/commit/3adb0b542450b123c1cef22126d054e183b21b24))

* Clarify multi-gpu usage ([`5a2da23`](https://github.com/embeddings-benchmark/mteb/commit/5a2da23c801d680f4782990d3af9d48ad20030cc))

* Fix splits ([`93f6f85`](https://github.com/embeddings-benchmark/mteb/commit/93f6f8557e63dc7db0e1a41c1e6599dc4b748a93))

* Improve Cust Model explanation ([`52c1fd8`](https://github.com/embeddings-benchmark/mteb/commit/52c1fd8d1f5eb07a2d0d322a1116d7e800782d4a))

* Add bs to Clustering test ([`4df0d2e`](https://github.com/embeddings-benchmark/mteb/commit/4df0d2ed7c42b180f268555b238cecd1beffec02))

* Rely on auto-conversion to tensor in score function ([`d8512f7`](https://github.com/embeddings-benchmark/mteb/commit/d8512f7d7dd7c3e71273fa948357ab76e5613689))

* Rely on standard encode kwargs only ([`4c1660e`](https://github.com/embeddings-benchmark/mteb/commit/4c1660e99b350269a757ab9b1d5a6d380a1f6475))

* Improve Cust Model explanation ([`23d758f`](https://github.com/embeddings-benchmark/mteb/commit/23d758fd73ddc80a918420d11d25d7ced7a2d008))

* Add bs to Clustering test ([`6e0c0d2`](https://github.com/embeddings-benchmark/mteb/commit/6e0c0d2e53556e4a3c0c345233a20ef11a661972))

* Rely on auto-conversion to tensor in score function ([`7ec4c57`](https://github.com/embeddings-benchmark/mteb/commit/7ec4c57687b95ecfc7e84f062e750239b400e5a0))

* Rely on standard encode kwargs only ([`2fad0f9`](https://github.com/embeddings-benchmark/mteb/commit/2fad0f950acb36a38c4595fa5f9421cc6080c8bc))

* Update README.md ([`d9aa70f`](https://github.com/embeddings-benchmark/mteb/commit/d9aa70ffaad035d86815e8cf57ca3c6877b2f471))

* Update README.md ([`2211f83`](https://github.com/embeddings-benchmark/mteb/commit/2211f830cbea9e13bef5220576fc061407a548d2))

* Simplify assertion ([`f7fcbc1`](https://github.com/embeddings-benchmark/mteb/commit/f7fcbc18e99eb9f827490a6e87a76a4fd8913de7))

* Default to false ([`d64f6c7`](https://github.com/embeddings-benchmark/mteb/commit/d64f6c7be7cda5a1c34bb41012008f20fdd46ebf))

* Add multi gpu eval to readme (#140)

update readme ([`1b1c9d3`](https://github.com/embeddings-benchmark/mteb/commit/1b1c9d319f9a51ffd50c76e34c67773fc0ac75da))

* Support Multi-node Evaluation (#132)

* styling

* USE_HF_DATASETS

* Support DRPES

* we use beir.datasets.data_loader_hf in case of non dist

* distributed fixes

* update run command

* cleanup

* .

* sugg

* ruff ([`0dd82a9`](https://github.com/embeddings-benchmark/mteb/commit/0dd82a9819d32f264a24dbc057f753efbf54e9d8))

* Add Chinese tasks (C-MTEB) (#134)

* add C_MTEB

* add C_MTEB

* rename MMarcoReranking

* rename MMarcoReranking

* Update mteb/tasks/Retrieval/CMTEBRetrieval.py

* Update README.md

* Allow custom encode functions

---------

Co-authored-by: shitao &lt;stxiao@bupt.edu.cn&gt;
Co-authored-by: Nouamane Tazi &lt;nouamane98@gmail.com&gt;
Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`071974a`](https://github.com/embeddings-benchmark/mteb/commit/071974a58e394332546a06a538c823440ee9ce73))

* Add Polish tasks (PL-MTEB) (#137)

* Add Polish tasks (PL-MTEB)

* Add Polish datasets to README

* Add newline

---------

Co-authored-by: rposwiata &lt;rposwiata@opi.org.pl&gt;
Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`2779344`](https://github.com/embeddings-benchmark/mteb/commit/277934453a5a82c57d3b173e3c52ce0f4d0c97a5))

* Add BEIR-PL datasets to MTEB (#121)

* Add BIER-PL benchmark

* Update README with BEIR-PL datasets

* Update names

* Add tasks to init to be visible during evaluation

---------

Co-authored-by: Konrad Wojtasik &lt;konrad.wojtasik@pwr.edu.pl&gt;
Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`5972c02`](https://github.com/embeddings-benchmark/mteb/commit/5972c0238b3d119ff0fa2822a678b5d9f17f1087))

* Replaced prints with logging (#133)

* Make sure that main score is added to bitext mining tasks

* Added scandinavian languages: da, no, sv

* merge upstream main

* fix: Replaced prints with logging statements

* chore: removed accidental commits ([`d7ca378`](https://github.com/embeddings-benchmark/mteb/commit/d7ca3784451873042fb8a3fc2cdf4406f1ab465a))

* add logging ([`6412a6a`](https://github.com/embeddings-benchmark/mteb/commit/6412a6ae9839636f3dafb9f4b1a35c2f2b22c76e))

* Merge pull request #131 from embeddings-benchmark/nouamane/quick-fixes

Code cleanup ([`4fb97d0`](https://github.com/embeddings-benchmark/mteb/commit/4fb97d0497a99a97833d813701fcd38ffbd05669))

* . ([`3ebb039`](https://github.com/embeddings-benchmark/mteb/commit/3ebb0399f6ea45b040a30bed426769a03bda264d))

* add eval_splits arg ([`c407c4b`](https://github.com/embeddings-benchmark/mteb/commit/c407c4b78fa74e2d494517e9b1921334f56dc82f))

* quick fixes ([`6c5a3fa`](https://github.com/embeddings-benchmark/mteb/commit/6c5a3fafa88529b5eee016404dcdddb7d5656a37))

* clean MTEB tasks ([`b276f1d`](https://github.com/embeddings-benchmark/mteb/commit/b276f1d49fe824069368f44a0aaa44e315162e65))

* clean args ([`9365755`](https://github.com/embeddings-benchmark/mteb/commit/9365755213a6fc39153d16d8c5c45d484da6075d))

* styling ([`dd02b48`](https://github.com/embeddings-benchmark/mteb/commit/dd02b48c58b4d33bea151420a78eec863fb7bffa))

* black ([`652d07c`](https://github.com/embeddings-benchmark/mteb/commit/652d07c70a0aa7e0380d412d5f0d6d744685d445))

* Set dev version ([`bf98c2c`](https://github.com/embeddings-benchmark/mteb/commit/bf98c2c33021141ce237a819e41be9371479bdac))


## v1.1.0 (2023-07-31)

### Unknown

* Release: 1.1.0 ([`80d0344`](https://github.com/embeddings-benchmark/mteb/commit/80d0344dfe93b8ef4114ae8b03aeb64032263fde))

* Bump version ID and update PyPI (#128)

Bump version ID and update PyPI after adding additional tasks. ([`4a4b54b`](https://github.com/embeddings-benchmark/mteb/commit/4a4b54b3ef22a39895dbc836fb7a2edb26508a94))

* Fix typo ([`33a3140`](https://github.com/embeddings-benchmark/mteb/commit/33a3140bea6255b6b1ecbee8f816a25f3326674e))

* Sort imports ([`ab2eef8`](https://github.com/embeddings-benchmark/mteb/commit/ab2eef85ec3641987dbddeb7bdf26d3ab2335707))

* Sort imports ([`3432374`](https://github.com/embeddings-benchmark/mteb/commit/34323741f0a9696be34b5b451319ed84534e839f))

* Raise error first ([`0b1bfd2`](https://github.com/embeddings-benchmark/mteb/commit/0b1bfd250ff1ca048d2c03026fa761bcbdf036b0))

* Added support for Scandinavian Languages (#124)

* Make sure that main score is added to bitext mining tasks

* Added scandinavian languages: da, no, sv

* Updated readme with scandinavian tasks

* Changes n samples for the nordic lang CLF

* Added scandinavian models to init

* Added error logs to gitignore

* fix import error

* fix dataset columns

* rename dataset columns

* remove swefaq

* fix: Added functionality to raise error

* fix: Updated names

* fix: Removed no as a language

* Added missing data transformation

* Fix spelling error ([`acb0f59`](https://github.com/embeddings-benchmark/mteb/commit/acb0f59435ee660c266490bfa1db22bd5f19d1d5))

* Install beir ([`c50b8ab`](https://github.com/embeddings-benchmark/mteb/commit/c50b8abd995ef1d2f7180f2de6fc5d3013901165))

* Update README.md ([`29ffedf`](https://github.com/embeddings-benchmark/mteb/commit/29ffedf77fe912400ed3b3558f08820dd5857b8f))

* ruff ([`6a58b5d`](https://github.com/embeddings-benchmark/mteb/commit/6a58b5d3b0c122d004de3f7476c1c169432208f5))

* Update README.md ([`5825536`](https://github.com/embeddings-benchmark/mteb/commit/582553693de507f338a720a0bc572147b8c6ef33))

* fix revision hash for TenKGnadClusteringP2P dataset

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`eb622f8`](https://github.com/embeddings-benchmark/mteb/commit/eb622f88e6700a10900a9732509aefe0ae8358b0))

* change dataset order for BlurbsClustering in README

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`f6e49ba`](https://github.com/embeddings-benchmark/mteb/commit/f6e49ba09a8c7a47008fef03fe834e3fa19d03e3))

* change dataset order for TenKGnadClustering in README

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`2a2c47f`](https://github.com/embeddings-benchmark/mteb/commit/2a2c47f6bbbed94308f55cde232168b33d12a50d))

* fix descriptions for German clustering datasets ([`30a966c`](https://github.com/embeddings-benchmark/mteb/commit/30a966c0a1294bd9d0c60e76ebb60c06477d8171))

* add German clustering tasks to README ([`62457e3`](https://github.com/embeddings-benchmark/mteb/commit/62457e311856d98360291376e67121b9954caf3c))

* update reference &amp; category for TenKGnad datasets ([`2174a47`](https://github.com/embeddings-benchmark/mteb/commit/2174a47e7f3d312cadbd708dc1eb65c90f6ef6e0))

* add German clustering tasks ([`ab469be`](https://github.com/embeddings-benchmark/mteb/commit/ab469be4ba5765639c090f72e98fb17aa6d07867))

* Allow abs path ([`b56528c`](https://github.com/embeddings-benchmark/mteb/commit/b56528cea68eeac169c5921537db4310c93642aa))

* Add @property annotation to description method of AbsTask ([`98b0443`](https://github.com/embeddings-benchmark/mteb/commit/98b0443b621ec40dd74296eabe6686d363288c65))

* fix typo ([`37a986b`](https://github.com/embeddings-benchmark/mteb/commit/37a986b13939cd15c13d611560b96acd2fa9897b))

* fix extend lang pairs ([`865dffc`](https://github.com/embeddings-benchmark/mteb/commit/865dffc73739c18b5fe4e38d079570c4bc032c1d))

* Fix clustering eval, black, isort ([`bc43665`](https://github.com/embeddings-benchmark/mteb/commit/bc43665504ab4241696c9983198d1240769dd2bc))

* Add &#39;auto&#39; to sklearn clustering, add test, fix warning ([`15ce352`](https://github.com/embeddings-benchmark/mteb/commit/15ce35239d8ab35829461c953e2505a0dc0b0613))

* Update MSMARCORetrieval.py ([`d913f56`](https://github.com/embeddings-benchmark/mteb/commit/d913f5606ceb6fa883c7134e12756334aa3e39e9))

* Revert to old split ([`1f3ff6e`](https://github.com/embeddings-benchmark/mteb/commit/1f3ff6e8f693af07c0f97eef401fe83230e8e1ec))

* Add wheel instruction ([`62fad9b`](https://github.com/embeddings-benchmark/mteb/commit/62fad9b42067893a1e175b9f1690967a7832e18e))

* Dev version ([`d988e48`](https://github.com/embeddings-benchmark/mteb/commit/d988e483e2af7dd96de40fa760f3b87dfcc86929))


## v1.0.2 (2023-03-28)

### Unknown

* Release: 1.0.2 ([`e189bae`](https://github.com/embeddings-benchmark/mteb/commit/e189baeb6a1bcd49e3fc7f9ebcbf76b2d37a0096))

* Add comment

Co-authored-by: Nouamane Tazi &lt;nouamane98@gmail.com&gt; ([`3e72ee8`](https://github.com/embeddings-benchmark/mteb/commit/3e72ee8b15883d2cfda4b658f08d8be33f86fd49))

* Fix naming ([`33f2db9`](https://github.com/embeddings-benchmark/mteb/commit/33f2db97a018e3408944e0f470ac6ff686107046))

* Cleaner logging &amp; tqdm usage ([`542d871`](https://github.com/embeddings-benchmark/mteb/commit/542d871746a99a051c4235627fa2d72f975fe5a1))

* Add kwargs ([`e0b801d`](https://github.com/embeddings-benchmark/mteb/commit/e0b801d4b481c15a244ba9fa5631db591f688cd7))

* Produce embeddings in one go ([`e88bcf2`](https://github.com/embeddings-benchmark/mteb/commit/e88bcf2345c4a07bf3adc0b21012e2652a7346ec))

* Fix naming ([`6c62f18`](https://github.com/embeddings-benchmark/mteb/commit/6c62f1885357ca5b16740907bd43507814767e87))

* Make inputs always List[str] &amp; call in one ([`bdeeedf`](https://github.com/embeddings-benchmark/mteb/commit/bdeeedf8d82506fb55682ba3d6c3083e68b4cae7))

* Fix SummEval description ([`0c2b1be`](https://github.com/embeddings-benchmark/mteb/commit/0c2b1befada16c515e295c8e57ba6b8b6bf0337d))

* fix SemmEval description

Unless I&#39;m missing something, I think the SemmEval description is incorrect---the dataset consists of summaries of news articles, not biomedical abstracts. ([`1ccc068`](https://github.com/embeddings-benchmark/mteb/commit/1ccc068a956bf72ab2b24cf3c73166dc222d7937))

* Clarify script for running all of MTEB English ([`9f72434`](https://github.com/embeddings-benchmark/mteb/commit/9f72434e0519eb2c5bfdacc4ef6a8c4679694a7a))

* Update run_mteb_english.py ([`6ff57d3`](https://github.com/embeddings-benchmark/mteb/commit/6ff57d3498cdaefec7658560e210df1faad047c3))

* Update run_mteb_english.py ([`7803eea`](https://github.com/embeddings-benchmark/mteb/commit/7803eea74d4a51691e71b3a4077ef8cd8dbb4051))

* Point to English benchmarking script ([`57f3371`](https://github.com/embeddings-benchmark/mteb/commit/57f3371e1692170711f1fb49043231fdd95fbbf7))

* Eexample script for benchmarking all of MTEB English ([`77e6b22`](https://github.com/embeddings-benchmark/mteb/commit/77e6b22ab164095c7de6ebdf1303f4db40796511))

* Clarify MSMARCO split ([`bbeada8`](https://github.com/embeddings-benchmark/mteb/commit/bbeada8e6018bcef1e1f74f28f9f12420f54b7b3))

* Allow re-merging ([`b0ce501`](https://github.com/embeddings-benchmark/mteb/commit/b0ce501202749b8aa70c8ac41462dd8a260fe2b9))

* Set dataset name; Sort imports ([`2a5a661`](https://github.com/embeddings-benchmark/mteb/commit/2a5a661fad2f2dab8558216ba1f773950a25d8ea))

* Standardize CQA merging script ([`5d5a2fb`](https://github.com/embeddings-benchmark/mteb/commit/5d5a2fbfe887ed9c360b863b67e8424f15fd96ae))

* Update merge_cqadupstack.py ([`b0304c1`](https://github.com/embeddings-benchmark/mteb/commit/b0304c16feb5a327a4772d9eddb7b3b52295b77e))

* Update README.md ([`8c60c22`](https://github.com/embeddings-benchmark/mteb/commit/8c60c22473960da38b9b560e824e53efa3c28496))

* Update README.md ([`6255449`](https://github.com/embeddings-benchmark/mteb/commit/625544968a53f62fe09655ce807f6d95f8da862f))

* Remove validation split ([`875a98e`](https://github.com/embeddings-benchmark/mteb/commit/875a98e50023686b476e608c983b8b639e462603))

* Remove validation set ([`b3f9585`](https://github.com/embeddings-benchmark/mteb/commit/b3f9585acc76150a1850c815bbf52f202feeeac3))

* Update ClassificationEvaluator.py ([`93b89b6`](https://github.com/embeddings-benchmark/mteb/commit/93b89b6824e0feb53df7f67a4f16737d37807a5a))

* Set dev version ([`8a0d6b1`](https://github.com/embeddings-benchmark/mteb/commit/8a0d6b1c1764f917b09bac454a994b0d870ca2c7))


## v1.0.1 (2022-11-29)

### Unknown

* Release: 1.0.1 ([`b9f423b`](https://github.com/embeddings-benchmark/mteb/commit/b9f423b25f9d054974d3aed6253e384409e18158))

* Delete mteb_diagram.png ([`76dc363`](https://github.com/embeddings-benchmark/mteb/commit/76dc3637ed2b49d0c182e0fff762a6d5cad37105))

* Deactivate beir ([`b263157`](https://github.com/embeddings-benchmark/mteb/commit/b2631578f674d89f2e4ae0634974f28c22a13408))

* Update BeIRTask.py ([`37b7b79`](https://github.com/embeddings-benchmark/mteb/commit/37b7b7915128dbc70e87ec5914058d48be00dfbe))

* Remove validation ([`6922840`](https://github.com/embeddings-benchmark/mteb/commit/69228402cde7f755a3d764dffc51fc8cb4b364eb))

* Fix typo ([`7247233`](https://github.com/embeddings-benchmark/mteb/commit/72472339fac9235a55ae67cea2aa596a4b83bcfc))

* Add files via upload ([`9d2bb67`](https://github.com/embeddings-benchmark/mteb/commit/9d2bb67914db80c9af654ee9c3d80e996622360f))

* Increment version &amp; use abslink ([`a792a65`](https://github.com/embeddings-benchmark/mteb/commit/a792a65da5645b5666436169ee6f167ec7904355))


## v1.0.0 (2022-10-17)

### Unknown

* Release: 1.0.0 ([`9c544a4`](https://github.com/embeddings-benchmark/mteb/commit/9c544a465f05628945ab1dccc67a05c5da90f53b))

* Add paper ([`b73457a`](https://github.com/embeddings-benchmark/mteb/commit/b73457a14e682225184b95fbcd589620991b5df4))

* Fix formatting ([`c523d16`](https://github.com/embeddings-benchmark/mteb/commit/c523d1600913698f816580657e024b4d736e66a3))

* print -&gt; logging ([`4f3a559`](https://github.com/embeddings-benchmark/mteb/commit/4f3a5590d3c2ab2dd925832eaff0324850887a6c))

* Do not ignore data scripts ([`891b455`](https://github.com/embeddings-benchmark/mteb/commit/891b455f1b8401ce11563e550647f6d67a6987b1))

* Reorganize scripts ([`e157bb0`](https://github.com/embeddings-benchmark/mteb/commit/e157bb035f9591af561613a2af0b31f5db98e2cd))

* Add release instructions &amp; dev suffix to version ([`164b9ae`](https://github.com/embeddings-benchmark/mteb/commit/164b9aeb41c725dfccba7c2ed29ddba187ad8502))


## v0.9.1 (2022-10-13)

### Unknown

* Release: 0.9.1 ([`5c438cc`](https://github.com/embeddings-benchmark/mteb/commit/5c438cc3b9efb0f5fbd35516e4f41aa3bcd9b5fc))

* Merge pull request #80 from embeddings-benchmark/Muennighoff-patch-5

Update STS22CrosslingualSTS.py ([`1459309`](https://github.com/embeddings-benchmark/mteb/commit/1459309b99ea9fcccfffa762301eac4e9565f651))

* Update installation ([`f96ee73`](https://github.com/embeddings-benchmark/mteb/commit/f96ee732c9c8a523361469f35cab371e0a75fa5f))

* Update SummEvalSummrization.py ([`d8f232d`](https://github.com/embeddings-benchmark/mteb/commit/d8f232dd46919ba238f7d5ba4800f72a91567f22))

* Update AmazonPolarityClassification.py ([`114b0e3`](https://github.com/embeddings-benchmark/mteb/commit/114b0e30bfe5c3f8a6e06714629b917d49e5aa56))

* Update STS22CrosslingualSTS.py ([`c8df727`](https://github.com/embeddings-benchmark/mteb/commit/c8df727013009eee7227c018e8de389d174e1606))

* Temporarily change README installation instruction ([`e53e77c`](https://github.com/embeddings-benchmark/mteb/commit/e53e77cb1c8bed06436228c58d8b6e51439905cd))

* Fix res keyword ([`769ac67`](https://github.com/embeddings-benchmark/mteb/commit/769ac6728fb9dd98e52f393f7b1b76cc43fc8d14))

* Update example to be visible for non-registered users ([`d4f75fc`](https://github.com/embeddings-benchmark/mteb/commit/d4f75fc3687673d0cba912f72d91c47b8c6dba92))

* Merge pull request #79 from Muennighoff/feature/leaderboardexp

Add leaderboard instructions ([`4d2683a`](https://github.com/embeddings-benchmark/mteb/commit/4d2683abaccd5fc7f1a63ca223403f09bd672966))

* Move meta script ([`7a8398f`](https://github.com/embeddings-benchmark/mteb/commit/7a8398f016bb5ed1a83b15910bec597aec533222))

* dataset_version -&gt; dataset_revision &amp; logging ([`fe34f84`](https://github.com/embeddings-benchmark/mteb/commit/fe34f84756b1269664bcc3828f3e9bd8707d4acc))

* Add leaderboard instructions ([`f325aca`](https://github.com/embeddings-benchmark/mteb/commit/f325acaf7a34b29b07ecd4e176faf4752832a3b5))

* Merge pull request #78 from embeddings-benchmark/feature/add-mteb-ds-name

Add ds name to res dict ([`53b763a`](https://github.com/embeddings-benchmark/mteb/commit/53b763adc3438eddf16736e1750b76d97b3422e2))

* Update MTEB.py ([`ae86e2f`](https://github.com/embeddings-benchmark/mteb/commit/ae86e2f5b07d97d5e3bb782d343861c29c01bfcd))

* Merge pull request #73 from Muennighoff/fix/cqadupstackbeir11

Fallback to old dataloader for cqadupstack ([`7791b41`](https://github.com/embeddings-benchmark/mteb/commit/7791b41a50884150a2d3f2463422d0b5f027c445))

* Merge pull request #77 from Muennighoff/fix/bcpc

Update init imports ([`865bf47`](https://github.com/embeddings-benchmark/mteb/commit/865bf473ae243082e531e785934e1170b32b3a93))

* Update init imports ([`39b7712`](https://github.com/embeddings-benchmark/mteb/commit/39b77124220acd940fd91e7878de94b9a89003d7))

* Merge pull request #76 from Muennighoff/fix/bcpc

BC -&gt; PC ([`82d3228`](https://github.com/embeddings-benchmark/mteb/commit/82d3228ac8b750558f8f9aa37d7a483fb6622acc))

* Merge branch &#39;main&#39; into fix/bcpc ([`f18c6df`](https://github.com/embeddings-benchmark/mteb/commit/f18c6dfa756cd2e9bcc2649eaccd339cf0089e13))

* BC -&gt; PC ([`7a430c2`](https://github.com/embeddings-benchmark/mteb/commit/7a430c21e20c2423f1fa38aea5330bfb1d4708d2))

* Merge pull request #75 from Muennighoff/feature/leaderboard

Add LB link ([`36dbd14`](https://github.com/embeddings-benchmark/mteb/commit/36dbd144cdead594233f85777e8d14e91b6f361f))

* Merge pull request #72 from Muennighoff/fix/revisions

Fix/revisions ([`4a8d3db`](https://github.com/embeddings-benchmark/mteb/commit/4a8d3db9c234dfc226019a01d9e07500b6977fd0))

* Merge pull request #74 from Muennighoff/fix/mteblogo

Update logo files ([`d939de6`](https://github.com/embeddings-benchmark/mteb/commit/d939de6dec4a8819c4eb13e07f1ee745158f8f46))

* Add LB link ([`6aeb7ed`](https://github.com/embeddings-benchmark/mteb/commit/6aeb7ed6e334c9b6278e9d93662200e58b4ce4a4))

* Update logo files ([`5bfb65a`](https://github.com/embeddings-benchmark/mteb/commit/5bfb65ab5553db2765b911467399c0fabd106eb1))

* Fallback to old dataloader for cqadupstack ([`262930e`](https://github.com/embeddings-benchmark/mteb/commit/262930e9069cad20c1f0c558448891cf2cf22fed))

* Add revision ([`488f1f7`](https://github.com/embeddings-benchmark/mteb/commit/488f1f7dbe8e9896fffa68d75f22497583939f98))

* Add revisions 2/2 ([`c8ba2b8`](https://github.com/embeddings-benchmark/mteb/commit/c8ba2b815b3c3e570393dc7fa7231b365726df35))

* Add revisions 1/2 ([`c75a503`](https://github.com/embeddings-benchmark/mteb/commit/c75a5033ace05eee466f5491f3404f41e4fa6517))

* Merge pull request #69 from Muennighoff/feature/custombeirmodel

Feature/custombeirmodel ([`da9ae9a`](https://github.com/embeddings-benchmark/mteb/commit/da9ae9a3c3800ab22c59ef65afbc0e20bbcc1e9f))

* BeIRModel -&gt; DRES ([`ff554bb`](https://github.com/embeddings-benchmark/mteb/commit/ff554bbc6dcc71666446a77d67b194bea6912153))

* Do not wrap 2x ([`255c416`](https://github.com/embeddings-benchmark/mteb/commit/255c4160fd0390eb44a2def08a5b892033286653))

* Adapt naming ([`3c8f672`](https://github.com/embeddings-benchmark/mteb/commit/3c8f672b803ed97f163fbc3e188c2877ce7e4d26))

* Add explanation of BeIRModel ([`3edad09`](https://github.com/embeddings-benchmark/mteb/commit/3edad09fb839ebfe9cabb556c7a2fc423f98bfc0))

* Merge pull request #68 from Muennighoff/feature/beirmrr

Add MRR ([`7a0993d`](https://github.com/embeddings-benchmark/mteb/commit/7a0993d9474082da56b7fc7986146cba564c832a))

* Allow custom BeIR model ([`cd5098b`](https://github.com/embeddings-benchmark/mteb/commit/cd5098b65605f832a3f58ed34513d577731935a2))

* Add MRR ([`6dbb97c`](https://github.com/embeddings-benchmark/mteb/commit/6dbb97cd4140ff243d455e47963b5eac77b9ea04))

* Merge pull request #67 from Muennighoff/fix/s2p

Fix categories ([`03ed576`](https://github.com/embeddings-benchmark/mteb/commit/03ed576dd2ad2b869591951c36c057e141930a66))

* Fix categories ([`08088d7`](https://github.com/embeddings-benchmark/mteb/commit/08088d7a9432502d802a22a860ffcd2c32906d0d))

* Update RedditClusteringP2P.py ([`77a1606`](https://github.com/embeddings-benchmark/mteb/commit/77a1606868f9bcae993637e5a35314ede7ec5786))

* Merge pull request #62 from Muennighoff/feature/hublinks

Feature/hublinks ([`4f04719`](https://github.com/embeddings-benchmark/mteb/commit/4f04719099f185b34a8c1b7adc736a7d9cae970c))

* Fix hub mistakes ([`02f9e6c`](https://github.com/embeddings-benchmark/mteb/commit/02f9e6c3b56b8c37976a273b34e562ae6e514b11))

* Merge branch &#39;feature/hublinks&#39; of https://github.com/Muennighoff/mteb into feature/hublinks ([`c98b9a6`](https://github.com/embeddings-benchmark/mteb/commit/c98b9a6a729b6234ca97d3631ea2bf69adcfcd5b))

* Add dataset stats ([`bbf2a82`](https://github.com/embeddings-benchmark/mteb/commit/bbf2a8253cf6fd01f196bc318bf36a25007e2217))

* Add desc ([`46078aa`](https://github.com/embeddings-benchmark/mteb/commit/46078aa614cef9ef988dfd296e7712e662a28ed0))

* Add desc ([`9ca92b0`](https://github.com/embeddings-benchmark/mteb/commit/9ca92b0243e20dce4ed579d9cdcf0af178760c75))

* Update MSMARCOv2Retrieval.py ([`f43cd1a`](https://github.com/embeddings-benchmark/mteb/commit/f43cd1adbe2df1ea6db9a0e26829d2b13a6d626c))

* Merge pull request #63 from embeddings-benchmark/Muennighoff-patch-4

Add desc ([`f93abff`](https://github.com/embeddings-benchmark/mteb/commit/f93abff68429ff04d8af8cd6eaf79429ac38ee78))

* Add desc ([`c972cc9`](https://github.com/embeddings-benchmark/mteb/commit/c972cc9781fa34d42d03f878df79cc26cbfde81e))

* Merge pull request #61 from embeddings-benchmark/fix/nolangs

Fix no langs ([`c15e1a7`](https://github.com/embeddings-benchmark/mteb/commit/c15e1a7663eb4a979e3ad07e2ef2f87286b87876))

* Merge branch &#39;main&#39; into feature/hublinks ([`c3990d6`](https://github.com/embeddings-benchmark/mteb/commit/c3990d6277f91ead4954ba5f5fc15a380f6e6563))

* Simplify ([`936eee2`](https://github.com/embeddings-benchmark/mteb/commit/936eee28fa453758e25666b728ff68e8c5e65fc9))

* Add Hub links &amp; descriptions ([`b8182bb`](https://github.com/embeddings-benchmark/mteb/commit/b8182bb51551a4ed29f71cac9fb8ee230dbc2473))

* Update MTEB.py ([`0be4a06`](https://github.com/embeddings-benchmark/mteb/commit/0be4a0678ad8252bae9e5e948cfde3d2150b0a46))

* Merge pull request #57 from embeddings-benchmark/Muennighoff-patch-2

Update README.md ([`1ebca84`](https://github.com/embeddings-benchmark/mteb/commit/1ebca84b9b78da579a2b7305a93d86047e2dba0b))

* Merge pull request #59 from embeddings-benchmark/Muennighoff-patch-3

Update README.md ([`3f53c85`](https://github.com/embeddings-benchmark/mteb/commit/3f53c85fc6a2e698e273dc7242737ecf46ebd14b))

* Update README.md ([`8097f31`](https://github.com/embeddings-benchmark/mteb/commit/8097f317aa96bc9de41fcda6b4eac673491b95e2))

* Update README.md ([`5b260a4`](https://github.com/embeddings-benchmark/mteb/commit/5b260a4fcf8bfb091609af00ec8ef8e60a3a6098))

* Merge pull request #56 from Muennighoff/feature/readmelinks

Add README Links &amp; Images ([`f473dbd`](https://github.com/embeddings-benchmark/mteb/commit/f473dbd457229bdbdead4a4c0a3d6e47f6d63e80))

* Center title ([`1341db7`](https://github.com/embeddings-benchmark/mteb/commit/1341db76c62b653c2c7dd923d8e1e581a2470cda))

* Center title ([`8b80471`](https://github.com/embeddings-benchmark/mteb/commit/8b80471a6528fd1497891a381f2e8248ecc6d9ad))

* Beautify ([`1ab8764`](https://github.com/embeddings-benchmark/mteb/commit/1ab87643e12b14bc30047f4bf366292400c97dca))

* Merge pull request #49 from Muennighoff/fix/cqadupstack

Fix CQADupstack ([`3a4dd84`](https://github.com/embeddings-benchmark/mteb/commit/3a4dd848206731c852545971704068a32a95d999))

* Merge pull request #50 from Muennighoff/fix/redditp2p

New RedditP2P Script ([`7bc547e`](https://github.com/embeddings-benchmark/mteb/commit/7bc547e279a4273e8e1d814234014523ec537b50))

* Merge pull request #52 from Muennighoff/fix/bucc

Default to 1-indexed gold ([`9aff7f2`](https://github.com/embeddings-benchmark/mteb/commit/9aff7f28404691939cb08cc8d2b9505091fbb468))

* Merge pull request #54 from embeddings-benchmark/Muennighoff-patch-1

Update MSMARCORetrieval.py ([`3951c41`](https://github.com/embeddings-benchmark/mteb/commit/3951c41bcff6b1258cfbb1bb145f02f90d9044a8))

* Update MSMARCORetrieval.py ([`6922be0`](https://github.com/embeddings-benchmark/mteb/commit/6922be0312533d8c1102a8c99ab4bd6a098bd4e3))

* Default to 1-indexed gold ([`f29e1fb`](https://github.com/embeddings-benchmark/mteb/commit/f29e1fb5877ffd05a245032291bdbd0bdaa18f41))

* New RedditP2P Script ([`f73b179`](https://github.com/embeddings-benchmark/mteb/commit/f73b179272339304ee19e5f1f24bc768644186a8))

* Fix split ([`e3ea40b`](https://github.com/embeddings-benchmark/mteb/commit/e3ea40b7c29a69ae7665a500102b64f6ad52d77d))

* Add CQADupStack subsets ([`a32c00b`](https://github.com/embeddings-benchmark/mteb/commit/a32c00b8a71780d34b0e5708b91407e3db011f24))

* Fix CQADupstack ([`a26229f`](https://github.com/embeddings-benchmark/mteb/commit/a26229fe68b8a7294c52b974145d4a6e2f4ff1b2))

* Merge pull request #46 from Muennighoff/fix/scidocs

Fix/scidocs ([`ea10703`](https://github.com/embeddings-benchmark/mteb/commit/ea1070366ea9a56973b8cd87e4b694d9f5c4a56c))

* Update README name ([`afddfd3`](https://github.com/embeddings-benchmark/mteb/commit/afddfd38f42e3a1035586d211efb9d872ff065e7))

* Merge pull request #45 from Muennighoff/feature/cachetestembs

Feature/cachetestembs ([`475420a`](https://github.com/embeddings-benchmark/mteb/commit/475420a9d64e1b7e85f6c4253c784b83ed60b68b))

* Merge pull request #44 from Muennighoff/fix/silentskip

Fix/silentskip ([`f7d6fd1`](https://github.com/embeddings-benchmark/mteb/commit/f7d6fd1ac6f17ea97efef7d760bb3334be6db7e2))

* Merge pull request #43 from Muennighoff/main

Add flag to overwrite results ([`ece590f`](https://github.com/embeddings-benchmark/mteb/commit/ece590feb66bc99d51407c3eebb7d7a6e4ac8693))

* Merge pull request #33 from Muennighoff/fix/summeval

Fix SummEval NaN scores ([`48586e2`](https://github.com/embeddings-benchmark/mteb/commit/48586e2dfbc14aa9b4d1718e4888ea631f1ae79e))

* Merge branch &#39;main&#39; into main ([`e986cd1`](https://github.com/embeddings-benchmark/mteb/commit/e986cd194a20c778628953d35829d2bb17e86bc2))

* Merge pull request #42 from Muennighoff/feature/versioning

Feature/versioning ([`1aeaede`](https://github.com/embeddings-benchmark/mteb/commit/1aeaeded0e41842643ec31e6697a37c634a94a79))

* Update mteb/evaluation/MTEB.py ([`23a473f`](https://github.com/embeddings-benchmark/mteb/commit/23a473f729391436af9995435731c71f1503cbfa))

* Rename SciDocs ([`edc2917`](https://github.com/embeddings-benchmark/mteb/commit/edc29176fe6458e4c667fe2f0a9b43374f047f16))

* Return test cache in all clf evaluators ([`309a867`](https://github.com/embeddings-benchmark/mteb/commit/309a867dda40ab0139537a1ed6eb9ce672e4e88b))

* Cache test embedding / exp for all clf evals ([`7dd867f`](https://github.com/embeddings-benchmark/mteb/commit/7dd867f343e016e54970273af259ff2d2feee34d))

* Add testcache ([`08cb352`](https://github.com/embeddings-benchmark/mteb/commit/08cb352f9ee5f1ac800e9afae027654ce9406372))

* Split into two lines ([`f756399`](https://github.com/embeddings-benchmark/mteb/commit/f756399b0c0b7865c90423e0f2f09a52eb3dbec5))

* Sort tasks ([`03658fa`](https://github.com/embeddings-benchmark/mteb/commit/03658fa5eef2b965489939f49499e4556f985b9c))

* Log known tasks ([`86f9cd6`](https://github.com/embeddings-benchmark/mteb/commit/86f9cd64655470e90885d540f6a7d0fcf11ed5b4))

* Log tasks not found ([`9ab0a7a`](https://github.com/embeddings-benchmark/mteb/commit/9ab0a7a33c69c2bbf8ee2f89b6dac3cd8b6660c7))

* Add flag to overwrite ([`529541d`](https://github.com/embeddings-benchmark/mteb/commit/529541d19bb043f236e753a5d31705e555a06d0a))

* Version mteb &amp; ds ([`78b90e9`](https://github.com/embeddings-benchmark/mteb/commit/78b90e994fddee2d4786639b6836c74cdb5dfa79))

* Formatting ([`67f6070`](https://github.com/embeddings-benchmark/mteb/commit/67f607031d12fac00dd26dc2a0a77882f62f56ee))

* Add versioning ([`fa852de`](https://github.com/embeddings-benchmark/mteb/commit/fa852de09e3546c3c231bacac147dc452a16dba2))

* Merge pull request #41 from Muennighoff/fix/sts22 ([`064e47c`](https://github.com/embeddings-benchmark/mteb/commit/064e47cedd40f3bb8c39f01b08eb55e28bcb9fd4))

* Rmv superfluous imports ([`7e8ee18`](https://github.com/embeddings-benchmark/mteb/commit/7e8ee183e0d984fc2fd34b816908b9a6634c8d23))

* Make revision optional ([`90afba5`](https://github.com/embeddings-benchmark/mteb/commit/90afba5af753d260926b48815ec3bec43766829f))

* Remove space ([`e0d22bc`](https://github.com/embeddings-benchmark/mteb/commit/e0d22bc8a6cf87049570929b76e67c9606bb54b4))

* Modify script to invert scores ([`9b9f43a`](https://github.com/embeddings-benchmark/mteb/commit/9b9f43a64154ecc5520c819da68d0efd4c2fcaed))

* Add revision to CL ([`5f68fda`](https://github.com/embeddings-benchmark/mteb/commit/5f68fda598d2ebd7093372194f2aa00426a23da8))

* Add revision kwarg ([`3448d1e`](https://github.com/embeddings-benchmark/mteb/commit/3448d1ea683527a7e2d6eb32f27068710a3cfb96))

* Merge pull request #26 from AmrMKayid/return-results ([`8f3242c`](https://github.com/embeddings-benchmark/mteb/commit/8f3242c39f469da53bc46817f06e5c91475d6f1b))

* Merge pull request #38 from Muennighoff/fix/seeds ([`720c597`](https://github.com/embeddings-benchmark/mteb/commit/720c597197470b91e9121390e1251367d69c290a))

* Update docs ([`dd4a1f2`](https://github.com/embeddings-benchmark/mteb/commit/dd4a1f257d2853ec7834d8b818367b9cc984f9d5))

* Merge pull request #37 from embeddings-benchmark/mindref

Fix Mind Reference ([`1834041`](https://github.com/embeddings-benchmark/mteb/commit/18340414855875013726497eacc47f28245e552f))

* Seed cuda ([`d33d748`](https://github.com/embeddings-benchmark/mteb/commit/d33d748bcd86c520add9b7cdcdf32d3a14502504))

* Merge pull request #35 from embeddings-benchmark/bootstrap-logs ([`3ff35c5`](https://github.com/embeddings-benchmark/mteb/commit/3ff35c5f23f8a3db92ccad49ae6c627387f4d69c))

* Update mteb/abstasks/AbsTaskClassification.py

Co-authored-by: Niklas Muennighoff &lt;n.muennighoff@gmail.com&gt; ([`9255249`](https://github.com/embeddings-benchmark/mteb/commit/9255249d50ae8eac126bb41459e40d70769c3c5c))

* Remove superfluous import ([`124bebe`](https://github.com/embeddings-benchmark/mteb/commit/124bebef96d8c432229f1ad00be0eebb6122c8f2))

* Remove superfluous comments ([`bf5f912`](https://github.com/embeddings-benchmark/mteb/commit/bf5f9121ca89854457edeb3ccefb0993146f2097))

* Add seed to task ([`acf8b1c`](https://github.com/embeddings-benchmark/mteb/commit/acf8b1cdd70dbaf0cbcc1d5dd230f5aca8a8c5c6))

* Add missing super calls ([`b32195e`](https://github.com/embeddings-benchmark/mteb/commit/b32195e5156a40b6cba0716158c10f5551856537))

* Set evaluation seeds ([`e69d40b`](https://github.com/embeddings-benchmark/mteb/commit/e69d40b0c683b55d517cb11ac31135c1c71c9e49))

* Set seeds ([`ef2985b`](https://github.com/embeddings-benchmark/mteb/commit/ef2985b88a9081e9bbad4500976847b05c0e3b1a))

* Fix Mind Reference

Two other notes:
- The renaming can create confusion as there exists a test set just that I assume we don&#39;t have the labels
- MIND uses AUC &amp; MRR &amp; NDCG scores, not MAP, see https://msnews.github.io/ ([`7ce4bb1`](https://github.com/embeddings-benchmark/mteb/commit/7ce4bb16db369258e1f88e4118baf247203a8e77))

* Update mteb/evaluation/evaluators/SummarizationEvaluator.py

Co-authored-by: Nouamane Tazi &lt;nouamane98@gmail.com&gt; ([`f667749`](https://github.com/embeddings-benchmark/mteb/commit/f66774973b614129feb446e0f0f26ee5e6a73a0d))

* Merge pull request #36 from embeddings-benchmark/mindsmall-test ([`6fc710b`](https://github.com/embeddings-benchmark/mteb/commit/6fc710b740d219e882203144b206085526581c9b))

* rename `validation`split to `test` ([`9c4d5c6`](https://github.com/embeddings-benchmark/mteb/commit/9c4d5c634301e953c5033809d36f2eee843e6616))

* styling ([`c66610e`](https://github.com/embeddings-benchmark/mteb/commit/c66610ee085f3b71b198d8d07317abecf6e5a8f3))

* add logs for classification bootstrap experiments ([`e4000e1`](https://github.com/embeddings-benchmark/mteb/commit/e4000e12e592ce079073ab3bb359b16f4a2d1196))

* Merge pull request #32 from Muennighoff/fixsplits ([`39d0926`](https://github.com/embeddings-benchmark/mteb/commit/39d0926ee3f2f094e843eab7cd16b762244c2183))

* Add consistent brackets ([`2cdd283`](https://github.com/embeddings-benchmark/mteb/commit/2cdd2836990ff7e2dab2f447acb9f3a9e5f8fbec))

* Remove debug leftovers ([`c674d0a`](https://github.com/embeddings-benchmark/mteb/commit/c674d0afce9b97428059cc1de9a6be8f2d6bbb96))

* Remove superfluous imports ([`68f7307`](https://github.com/embeddings-benchmark/mteb/commit/68f730773bdd2e4d12d0b8c5a410ad549e64170b))

* Skip samples with no variance ([`d39be65`](https://github.com/embeddings-benchmark/mteb/commit/d39be6541e7f95ce0f4bc729f3865493974b6c38))

* Drop nans ([`20c22a9`](https://github.com/embeddings-benchmark/mteb/commit/20c22a919ae07314e3f93f2ddd808e87b0c7dbff))

* Fix BEIR splits ([`752d49f`](https://github.com/embeddings-benchmark/mteb/commit/752d49fdd81f967bceb78656e9d1f27ce7efd539))

* Fix splits ([`07bea18`](https://github.com/embeddings-benchmark/mteb/commit/07bea1820d1a28ca0ac677601d9e622731fc4c0a))

* Merge branch &#39;main&#39; into return-results ([`314e5d7`](https://github.com/embeddings-benchmark/mteb/commit/314e5d72ac43233f656dd21c10202cc6ffef4602))

* Merge pull request #30 from embeddings-benchmark:selected_tasks

fix printing selected tasks for evaluation ([`f1cab40`](https://github.com/embeddings-benchmark/mteb/commit/f1cab400cb0c41a7cee83613d4e7927ac07fb71e))

* fix printing selected tasks for evaluation ([`ba0dd76`](https://github.com/embeddings-benchmark/mteb/commit/ba0dd767007cc5f4a068d479adbbf46c720b2b72))

* Merge pull request #29 from cycycc/fix-sickr-hf-hub-name ([`cb87c7a`](https://github.com/embeddings-benchmark/mteb/commit/cb87c7a388baa6461433e0dff3f47c18358b8a7d))

* fix sick-r huggingface hub name ([`2ea195a`](https://github.com/embeddings-benchmark/mteb/commit/2ea195a0fb7dd3d050f940240a8b3484c6b2cb64))

* Update mteb/evaluation/MTEB.py

Co-authored-by: holidaydrien &lt;adrien.morisot@gmail.com&gt; ([`a4d952b`](https://github.com/embeddings-benchmark/mteb/commit/a4d952b0ede70a621ea8c1e8bf393fc2b3d91109))

* Update mteb/evaluation/MTEB.py

Co-authored-by: holidaydrien &lt;adrien.morisot@gmail.com&gt; ([`c4acb76`](https://github.com/embeddings-benchmark/mteb/commit/c4acb76fdb9d06decad58b0a6fab1f4b0b4ddd01))

* Returning Evaluation results ([`3d60490`](https://github.com/embeddings-benchmark/mteb/commit/3d60490485bbc9d9e10c9d18ce15b2bfb444ed47))

* Merge pull request #18 from Muennighoff/evalfix ([`4dabbaf`](https://github.com/embeddings-benchmark/mteb/commit/4dabbaf94863e4f78e012481899bf2f7952f263b))

* Merge pull request #19 from Muennighoff/patch-2 ([`9e56ad3`](https://github.com/embeddings-benchmark/mteb/commit/9e56ad3d4d8089373a2c97cf85113b7c9f75f3fd))

* Merge pull request #20 from Muennighoff/updatemainscores ([`a0fbd83`](https://github.com/embeddings-benchmark/mteb/commit/a0fbd835062c202e91e76d82f234a7ef947efa9c))

* Update to ndcg_at_10 ([`8d010d0`](https://github.com/embeddings-benchmark/mteb/commit/8d010d006cfe48f463618b929c2af9cf2365f92f))

* Update main scores ([`c0e773a`](https://github.com/embeddings-benchmark/mteb/commit/c0e773a3ce04558c15ac27f660f05c0620efbc75))

* Update README.md ([`8b495b6`](https://github.com/embeddings-benchmark/mteb/commit/8b495b626a61564efa9ad9a3411890e1d45d35d1))

* Fix task splits ([`1755356`](https://github.com/embeddings-benchmark/mteb/commit/17553566046ade6d627266f4ad1ca06e6e615dc1))

* Merge pull request #15 from Muennighoff/mainscorefix ([`4b5fe2b`](https://github.com/embeddings-benchmark/mteb/commit/4b5fe2b58463ff0ae22ce6ed0e7707ba2c3a5c09))

* Fix monolingual mainscore ([`61647df`](https://github.com/embeddings-benchmark/mteb/commit/61647dff0e1cd80c0e3cc5bd80ac397c48f1c89b))

* Fix main score warning multilingual ([`831a218`](https://github.com/embeddings-benchmark/mteb/commit/831a218407caee59bb83af78f8b832d8c3af830c))

* Merge pull request #14 from Muennighoff/patch-1 ([`6055ecc`](https://github.com/embeddings-benchmark/mteb/commit/6055ecc90b0b093337530cb6f957ae12fdabb546))

* Fix task language example ([`115c280`](https://github.com/embeddings-benchmark/mteb/commit/115c28081f7434c8131548c79ba7b3d2201f83fc))

* styling ([`2ff07d2`](https://github.com/embeddings-benchmark/mteb/commit/2ff07d22932e697bbc84d745c46a4a7b1c34481f))

* update example ([`b581d00`](https://github.com/embeddings-benchmark/mteb/commit/b581d00bc384f5623a970b146a9614f93115b251))

* we can now select all tasks of a specific language ([`b36e58c`](https://github.com/embeddings-benchmark/mteb/commit/b36e58ca198c7b92e508fbec516ebd12dd7980b4))

* update test ([`53d123e`](https://github.com/embeddings-benchmark/mteb/commit/53d123e12465973d2b2e9670939f9134da7ae531))

* keep only langs defined in task&#39;s description when loading ([`efa189f`](https://github.com/embeddings-benchmark/mteb/commit/efa189fc29c0f2524088ab9c140e4978c25b355f))

* better prints for multilingual and crosslingual evaluation ([`5b86950`](https://github.com/embeddings-benchmark/mteb/commit/5b869501994121bdcb978ec7503ef91dd3473480))

* styling ([`8fd8fb0`](https://github.com/embeddings-benchmark/mteb/commit/8fd8fb06847b56d4d4c1f8d8ab5e316668bb7bdb))

* move scripts to respective folders ([`028ed3e`](https://github.com/embeddings-benchmark/mteb/commit/028ed3ec673142735f666e36fe05a13f0d28258a))

* Update gitignore ([`a3cee03`](https://github.com/embeddings-benchmark/mteb/commit/a3cee0313075b85be55e79136156010bb2fc218b))

* update setup.py ([`89aaa43`](https://github.com/embeddings-benchmark/mteb/commit/89aaa43c35cb355867cffb91a18e09b1ee107fe5))

* update setup ([`2645323`](https://github.com/embeddings-benchmark/mteb/commit/26453234ccceac7258bb15ef714e78bf3f2810c9))

* update setup.cfg ([`bc5ec1d`](https://github.com/embeddings-benchmark/mteb/commit/bc5ec1dbd5a521c2e8cf8245b22e2a3dafd920c9))

* Create first pip version ([`210d012`](https://github.com/embeddings-benchmark/mteb/commit/210d012b595e0d38867b7be66ebed3e0de78cb84))

* make default evaluation for classification 10 experiments each using 8 samples per label ([`b062405`](https://github.com/embeddings-benchmark/mteb/commit/b0624055f1f923321bd2d115e93dc3d7517ff297))

* use seed from init arg ([`f58f8da`](https://github.com/embeddings-benchmark/mteb/commit/f58f8da29921d143d7ba4b945c2391e7da296b62))

* styling ([`4d1bd09`](https://github.com/embeddings-benchmark/mteb/commit/4d1bd092a955174f6cdb8b7d69ad783eaf1cab98))

* add error message when trying to load beir ([`a3d58f3`](https://github.com/embeddings-benchmark/mteb/commit/a3d58f37dac2307e3c05b71c62975ec85c64f6f3))

* add argument to specify error logs path ([`d6cef16`](https://github.com/embeddings-benchmark/mteb/commit/d6cef16190fc2fbf52ab0a8416524ec0f20a8bc4))

* make beir an optional package ([`5bcee12`](https://github.com/embeddings-benchmark/mteb/commit/5bcee120858973d5eac8736c2e489c02714c78a3))

* quick modifications ([`d774ce6`](https://github.com/embeddings-benchmark/mteb/commit/d774ce6e11819745db39a347792d8355a53eb70a))

* add example ([`21fc624`](https://github.com/embeddings-benchmark/mteb/commit/21fc6249b19105ec8daec767c49fafbefa7d1dfc))

* make beir optional dependency ([`fdd922a`](https://github.com/embeddings-benchmark/mteb/commit/fdd922a52450c56196bb67f231a28d5f2c92a44f))

* Smaller fixes in Classification task ([`c6eda26`](https://github.com/embeddings-benchmark/mteb/commit/c6eda2694204178ed3757886c5aab14bcd69a178))

* update available tasks ([`0923e50`](https://github.com/embeddings-benchmark/mteb/commit/0923e5067853c2e067cbb5ef8fe2e56e90cd2d8e))

* update available tasks ([`e192823`](https://github.com/embeddings-benchmark/mteb/commit/e192823ecb31c277a2abab49749d60d14ced2e3e))

* add evaluation time to final scores ([`9a1ca7d`](https://github.com/embeddings-benchmark/mteb/commit/9a1ca7d1af8ed1f6baa7c7ade6b384e44ea054a8))

* quick fix loading beir task ([`8e46cc8`](https://github.com/embeddings-benchmark/mteb/commit/8e46cc8be633f6f1fece72028b0e82b33fde977a))

* add available tasks ([`b7a1987`](https://github.com/embeddings-benchmark/mteb/commit/b7a1987c3636055cdc9fa2babe2131d14295f0df))

* Merge pull request #11 from embeddings-benchmark/summarization ([`bdb2691`](https://github.com/embeddings-benchmark/mteb/commit/bdb26915d476c02e7688131ca671198e15b0a53e))

* add more scores to summarization evaluator ([`12ae05f`](https://github.com/embeddings-benchmark/mteb/commit/12ae05f012ddde96fa47b8656b3bf6560bc58123))

* add SummEval task ([`3ba3e65`](https://github.com/embeddings-benchmark/mteb/commit/3ba3e65f1e51188854c92c204c01ad8025d08cf2))

* add Summarization abstract task ([`f2b0e53`](https://github.com/embeddings-benchmark/mteb/commit/f2b0e533791f1aba83284b8e2c472201a68412d1))

* add specifying language for task example ([`cdf1f18`](https://github.com/embeddings-benchmark/mteb/commit/cdf1f18224cba9bca4d239407184e8f01fa124c9))

* fix bitext mining evaluation ([`073a254`](https://github.com/embeddings-benchmark/mteb/commit/073a254c3e6ee48129686505d3cea135e7cce50d))

* update README ([`3b30e9b`](https://github.com/embeddings-benchmark/mteb/commit/3b30e9b23b0dd1136865baa41f1e4e5bcb834832))

* update README ([`529ec6b`](https://github.com/embeddings-benchmark/mteb/commit/529ec6bd27f2c64528c95ae13033ed788a75c5b7))

* add --available_tasks flag to CLI ([`de97d9a`](https://github.com/embeddings-benchmark/mteb/commit/de97d9ac3d72596476dece5507e199d122be333e))

* styling ([`324b94c`](https://github.com/embeddings-benchmark/mteb/commit/324b94c30f1c208de841b37c9a62bf261c3ba5f6))

* fix missing params eval_splits in load_data ([`ecb9d12`](https://github.com/embeddings-benchmark/mteb/commit/ecb9d12f234ccec4a4ddbfd55636ae7338590daa))

* CLI quick fixes ([`693bffa`](https://github.com/embeddings-benchmark/mteb/commit/693bffaebd485ea68b4d79eb5232fd3035d7962b))

* Merge branch &#39;main&#39; of https://github.com/embeddings-benchmark/mteb-draft into main ([`bba225d`](https://github.com/embeddings-benchmark/mteb/commit/bba225d9811aecbe4050567387ed1928b164f71a))

* quick fixes ([`2c01099`](https://github.com/embeddings-benchmark/mteb/commit/2c01099cf355e261684c34fcdd97a2bc33fb0110))

* styling ([`75d0449`](https://github.com/embeddings-benchmark/mteb/commit/75d0449844a06e112a38a2d51c79a55ae2ac4f46))

* fix eval_splits loading using beir ([`26ec6b9`](https://github.com/embeddings-benchmark/mteb/commit/26ec6b9f75f40fd7118b0e001db3c7352bec446e))

* capture errors instead of failing ([`c6aafa4`](https://github.com/embeddings-benchmark/mteb/commit/c6aafa445e978fdeb6c2d754f17e27c5c727b3b1))

* quick fixes ([`8a7e3ec`](https://github.com/embeddings-benchmark/mteb/commit/8a7e3ec49041768fdbec114f17e447450b52a247))

* update BeIRModel ([`e8b5ff9`](https://github.com/embeddings-benchmark/mteb/commit/e8b5ff952bd6315433fdea926e25768612ae5a7f))

* load data and free it after each task evaluation ([`aa467f2`](https://github.com/embeddings-benchmark/mteb/commit/aa467f2534d018a3af302c4e2159789cf524bc31))

* update reqs ([`6005c10`](https://github.com/embeddings-benchmark/mteb/commit/6005c10509097c1749a72ef44df560a8cdef2e9d))

* fixing beir imports ([`5d74d42`](https://github.com/embeddings-benchmark/mteb/commit/5d74d423df169d03f732b4368ab36b59ede51a48))

* Merge pull request #10 from embeddings-benchmark/optimisation ([`2b6caf2`](https://github.com/embeddings-benchmark/mteb/commit/2b6caf216918c0b39681b83ff5fff003a99643f3))

* add multiproc test ([`fe8b963`](https://github.com/embeddings-benchmark/mteb/commit/fe8b9633098aacb785f8e331875f5eebee3038e1))

* update BitextMining main scores ([`3b0f912`](https://github.com/embeddings-benchmark/mteb/commit/3b0f91257c9eaaf5b7e4f15e248e48c0c5192e33))

* support distributed evaluation for IR 🥳 ([`5e91971`](https://github.com/embeddings-benchmark/mteb/commit/5e91971c809fade4b9f9cdd5c4217e50cb666e14))

* remove &#34;train&#34; from eval_splits ([`6da5ed1`](https://github.com/embeddings-benchmark/mteb/commit/6da5ed1846a2a572dbf65ae617efc3301cb9370b))

* gather all nodes outputs in CPU after distributed computation ([`5eb3661`](https://github.com/embeddings-benchmark/mteb/commit/5eb366156c203780a337b24551c0ae1103d09952))

* support DRPES for Parallel IR evaluation ([`36962e9`](https://github.com/embeddings-benchmark/mteb/commit/36962e94bf6970ecf073d9fbc98539d57eb781c7))

* quick fix ([`6e0e6bd`](https://github.com/embeddings-benchmark/mteb/commit/6e0e6bd01645a20142d4a1bc464e9bd18e2d5d6e))

* set logistic regression default max_iter to 200 ([`8963b83`](https://github.com/embeddings-benchmark/mteb/commit/8963b8315fc1c3efcce3fba0e3560749fd6992f5))

* add evaluators logs 📜 ([`e9d326f`](https://github.com/embeddings-benchmark/mteb/commit/e9d326f190100f265489123f0e53d06a2c57d9b0))

* make style ([`ab8f13e`](https://github.com/embeddings-benchmark/mteb/commit/ab8f13ec5b1045acca2f369a6798f9c767cc01d0))

* add Makefile and better styling tools ✨ ([`156e828`](https://github.com/embeddings-benchmark/mteb/commit/156e82814ab1b8e90ae5e643fd31559fb60e8bda))

* dataloading moved from __init__ to run ([`c2b7901`](https://github.com/embeddings-benchmark/mteb/commit/c2b7901bcf632ea325124ec69519a0dbdf51d98f))

* Merge pull request #8 from embeddings-benchmark/beir-integration

Beir integration ([`f7f2426`](https://github.com/embeddings-benchmark/mteb/commit/f7f2426180727a65637ec12f1230eb3a67744627))

* Merge branch &#39;main&#39; into beir-integration ([`af12b49`](https://github.com/embeddings-benchmark/mteb/commit/af12b49e43539d4c8e475e457996206b64425ab7))

* Merge pull request #9 from embeddings-benchmark/display

Display ([`11e5758`](https://github.com/embeddings-benchmark/mteb/commit/11e5758c82339fafd8274659c878d808983bec36))

* fixes ([`8902f59`](https://github.com/embeddings-benchmark/mteb/commit/8902f590476b13b85485fca4b38f235e8338d123))

* fixes ([`a394cc2`](https://github.com/embeddings-benchmark/mteb/commit/a394cc217f35962aebc425c7ed12f9ba456ca778))

* fixes+black ([`b0527a8`](https://github.com/embeddings-benchmark/mteb/commit/b0527a81c3f088644c79b9cc5977760e32ade7e5))

* beautiful task display ([`0ff2db2`](https://github.com/embeddings-benchmark/mteb/commit/0ff2db24eb5bda300aba7d175a07397418d52ca5))

* rich library ([`27cd4cb`](https://github.com/embeddings-benchmark/mteb/commit/27cd4cb00018f78c06183aae145a61091c6ad041))

* datasets ([`895c23d`](https://github.com/embeddings-benchmark/mteb/commit/895c23d66a5f6330899c8072f4e8bb44e67d52d2))

* fever ([`0724070`](https://github.com/embeddings-benchmark/mteb/commit/07240708fe30cebcf65a27b548843372a14b9a74))

* quora ([`43b93e5`](https://github.com/embeddings-benchmark/mteb/commit/43b93e5a0f2b645359b5a84858edd33da2c90abb))

* dbpedia ([`50d6700`](https://github.com/embeddings-benchmark/mteb/commit/50d67005e826c1fd29731e02f8e5dd2210d44bfb))

* climatefever ([`e506637`](https://github.com/embeddings-benchmark/mteb/commit/e5066373cbd079f8683a4500dbba0b4b26f96d2a))

* cqadupstack ([`217009f`](https://github.com/embeddings-benchmark/mteb/commit/217009faa2e12ca88c7f066101e03151b6b641f1))

* arguana ([`019b2b7`](https://github.com/embeddings-benchmark/mteb/commit/019b2b72e46e8d12c9ecd51e80ac02e289a098ae))

* beir retrieval ([`e52171b`](https://github.com/embeddings-benchmark/mteb/commit/e52171bb685debbc026a23957104c0be326ef479))

* only save if output_folder argument is specified ([`2e1eb24`](https://github.com/embeddings-benchmark/mteb/commit/2e1eb24d9b624c507e878b42a9f013a229c999fa))

* Update python-package.yml ([`6c32b6b`](https://github.com/embeddings-benchmark/mteb/commit/6c32b6bdc0bddbe9ab6e4aea2a1fa9f911840136))

* all tests are passing now ✅ ([`6c41b75`](https://github.com/embeddings-benchmark/mteb/commit/6c41b753311d8b985605fca60937eef8ce2b046f))

* Create python-package.yml ([`3cce88f`](https://github.com/embeddings-benchmark/mteb/commit/3cce88f5e7d87ddf0164e7f5854e587b2c8b2fa8))

* Merge pull request #6 from embeddings-benchmark/testing ([`06bd1df`](https://github.com/embeddings-benchmark/mteb/commit/06bd1df888689ea6b9133ff81c9bf03ce0387452))

* Merge branch &#39;main&#39; into testing ([`5226907`](https://github.com/embeddings-benchmark/mteb/commit/522690780f249b224895810ba30f0996041fd909))

* normalize STS scores ([`6f98396`](https://github.com/embeddings-benchmark/mteb/commit/6f983964eb721c55cff4dc9472d92550275c7cee))

* normalize score names ([`6db134f`](https://github.com/embeddings-benchmark/mteb/commit/6db134f62820ced9b9a9d3bb04504664b695e51c))

* format @k scores ([`dcb77a0`](https://github.com/embeddings-benchmark/mteb/commit/dcb77a0f96966acb62dc7aec7617d263ddc72be0))

* rename CrossLingual to Crosslingual ([`3af3b4f`](https://github.com/embeddings-benchmark/mteb/commit/3af3b4f88f1b0c9f70d52fd80043cb2cdee8d2e5))

* remove train split from evaluation splits ([`6317bb6`](https://github.com/embeddings-benchmark/mteb/commit/6317bb64c3ae1e25dc5c9e7f23846ba6a11d9b4b))

* bug fix ([`ba8c906`](https://github.com/embeddings-benchmark/mteb/commit/ba8c9064d6b70307e2f9d33e7b82315f03bf8d08))

* calculate AP only in binary classification ([`bc293ca`](https://github.com/embeddings-benchmark/mteb/commit/bc293cad20a24e719144fc71cdd67d41b5ace052))

* add kwargs and batch_size to evaluate funcs ([`7926d3c`](https://github.com/embeddings-benchmark/mteb/commit/7926d3cfe7187742cd2ebf361dfe4ef51df56bd8))

* update main scores for some tasks ([`099a32b`](https://github.com/embeddings-benchmark/mteb/commit/099a32b995e9d45ffe58554b63e4389f2522db36))

* add limit argument to limit evaluation data ([`92e5d09`](https://github.com/embeddings-benchmark/mteb/commit/92e5d098dea013439a63c22131d36b2ee640b568))

* add test for PairClassificationEvaluator ([`ecffd35`](https://github.com/embeddings-benchmark/mteb/commit/ecffd3536c26b26819c689bd9e8b35f37c881ff7))

* use evaluators.PairClassificationEvaluator instead of sent-formers BinaryClassificationEvaluator ([`9ffdf2b`](https://github.com/embeddings-benchmark/mteb/commit/9ffdf2b0fc9e1921cfb6680cd27b0d811bfeba43))

* reformatting ([`ca25e17`](https://github.com/embeddings-benchmark/mteb/commit/ca25e17b4c8df0c6d9e56e5297f515f55ee8a2ce))

* add test for RerankingEvaluator ([`9646892`](https://github.com/embeddings-benchmark/mteb/commit/9646892bdb4cddda3448b554343b0a12ffaa3348))

* reformat RerankingEvaluator ([`3ce99cf`](https://github.com/embeddings-benchmark/mteb/commit/3ce99cfa242a78436623ceabfeb01119d03b398d))

* more docs ([`8d07d59`](https://github.com/embeddings-benchmark/mteb/commit/8d07d590000c040886f792d3e959db68a844b9a0))

* tests folder ([`9cd9dc2`](https://github.com/embeddings-benchmark/mteb/commit/9cd9dc2bafd3ab518af3caf85df8b8b4cccd91f4))

* add test_RetrievalEvaluator ([`5236588`](https://github.com/embeddings-benchmark/mteb/commit/52365886d4515c15f8c4bdb0e49e4d1e1888862b))

* more docs ([`06ff3d1`](https://github.com/embeddings-benchmark/mteb/commit/06ff3d10e3ead80b6162e3d25f30cb93da92925a))

* add AP score to ClassificationEvaluator ([`c950ce8`](https://github.com/embeddings-benchmark/mteb/commit/c950ce89e1d88a56c1d2d7cda6ad90e93ff344f7))

* add nDCG score to RerankingEvaluator ([`e4170c8`](https://github.com/embeddings-benchmark/mteb/commit/e4170c83246ec729d3f6be46aababbe4fbc98b51))

* Merge pull request #5 from embeddings-benchmark:update-reranking

Support multiple queries in Reranking tasks ([`cf51493`](https://github.com/embeddings-benchmark/mteb/commit/cf514935c03040edb2bda00adb786817cb413777))

* quick fix ([`0d133e1`](https://github.com/embeddings-benchmark/mteb/commit/0d133e14091d47318669965d5d94c2e0e19def08))

* use max cross similarity in case of multiple queries ([`3f80a70`](https://github.com/embeddings-benchmark/mteb/commit/3f80a703bbb198998644398a4977991d001b0b43))

* support multiple queries in Reranking tasks ([`47f871f`](https://github.com/embeddings-benchmark/mteb/commit/47f871f8a036b82491e6e4315ed4410ccf415311))

* bug fixes ([`a3dc4f6`](https://github.com/embeddings-benchmark/mteb/commit/a3dc4f6077321ec55aa3fae8a038bd86620cf288))

* rename binary classification to pair classification ([`63374fe`](https://github.com/embeddings-benchmark/mteb/commit/63374fe9456ee27d3fa24ff9a5bb6b906812cdae))

* rename available_splits to eval_splits ([`04b9f55`](https://github.com/embeddings-benchmark/mteb/commit/04b9f55ec334f100d018b3d3c279a2e1899419c4))

* rename available_langs to eval_langs ([`99ad04c`](https://github.com/embeddings-benchmark/mteb/commit/99ad04ca2aa37238c719c3f996dfbb4de18f1f91))

* minor fixes ([`c2307ef`](https://github.com/embeddings-benchmark/mteb/commit/c2307ef2201b176ac5f11f6f43dbc39e97aae225))

* Merge pull request #4 from embeddings-benchmark/packaging ([`297560d`](https://github.com/embeddings-benchmark/mteb/commit/297560d99438e2f06eac0dbcdc75d97f4a817a50))

* quick fix bug ([`fc8ea9f`](https://github.com/embeddings-benchmark/mteb/commit/fc8ea9fa30f0cdb08e93bccc7b5e0395f278a4e7))

* report stderr in AbsTaskClassification in case of bootstrapping ([`d3723a5`](https://github.com/embeddings-benchmark/mteb/commit/d3723a591ec28f242e2ee4c74af940573e19288a))

* add STS22CrosslingualSTS ([`87f92e5`](https://github.com/embeddings-benchmark/mteb/commit/87f92e56ec947a702ebc7ad4a9d2deb191cba192))

* add MindSmallReranking ([`940642a`](https://github.com/embeddings-benchmark/mteb/commit/940642aac772ffc293ca5a340d1967e8a2314eea))

* precision recall f1 bitext evaluator ([`4f4a9e2`](https://github.com/embeddings-benchmark/mteb/commit/4f4a9e23df217675463bce42259a74a16f7cc4ea))

* korean to sts17 ([`4767300`](https://github.com/embeddings-benchmark/mteb/commit/4767300c3134176477a6089511d4a178d10f1e42))

* quick fix RetrievalEvaluator ([`afb574a`](https://github.com/embeddings-benchmark/mteb/commit/afb574a60d2fe26eb429d7f332f980583c3223bc))

* update README ([`db6edde`](https://github.com/embeddings-benchmark/mteb/commit/db6edde975eededec4779f6e8902315a160acb84))

* update example ([`d363a49`](https://github.com/embeddings-benchmark/mteb/commit/d363a493969da000cccfe104585fe25f44c5d63e))

* remove useless import ([`4a8966f`](https://github.com/embeddings-benchmark/mteb/commit/4a8966fbef42b9c21d4d465cd53ae95786606f8c))

* fix cmd.py arguments ([`2b17b4a`](https://github.com/embeddings-benchmark/mteb/commit/2b17b4a1552e71f298e23cbaea461ecc1efe4e17))

* add kwargs where needed ([`30f0efd`](https://github.com/embeddings-benchmark/mteb/commit/30f0efdee1ccd497a7622bd2428b7e02746e3438))

* add cli script ([`5a77900`](https://github.com/embeddings-benchmark/mteb/commit/5a77900e080a46833f0ca9792ef17d166195ce4c))

* adopt pbr packaging ([`c2fc3c1`](https://github.com/embeddings-benchmark/mteb/commit/c2fc3c1d3eccc247be7a9484ed880c9b2d43f25b))

* quick fixes ([`f5d3287`](https://github.com/embeddings-benchmark/mteb/commit/f5d32878b361dc4adeb1329e4938f7415af19533))

* rename kNNClassification to Classification ([`44ceb4a`](https://github.com/embeddings-benchmark/mteb/commit/44ceb4a6d87919d79648357ee083c007233adf2c))

* add bootstrap parameters to AbsTaskKNNClassification ([`8ecf9d4`](https://github.com/embeddings-benchmark/mteb/commit/8ecf9d49f17dc0a3e62fc5315d381c25e73e3209))

* add EmotionClassification ([`e03db39`](https://github.com/embeddings-benchmark/mteb/commit/e03db3942f9a3043cdbc376be460a17ad7b8d96b))

* add TweetSentimentExtractionClassification ([`5c7ef5c`](https://github.com/embeddings-benchmark/mteb/commit/5c7ef5c30fb432f1ad1579deb598eada8051abbb))

* add ToxicConversationsClassification ([`94bfb4f`](https://github.com/embeddings-benchmark/mteb/commit/94bfb4f71a7913e0a15738fd0ef5037b5874f24f))

* add AmazonCounterfactualClassification ([`4052ae1`](https://github.com/embeddings-benchmark/mteb/commit/4052ae1456d01b4e23681dc1dc0dd47efcee8fa5))

* add ImdbClassification task ([`eb70842`](https://github.com/embeddings-benchmark/mteb/commit/eb70842cacb635547f24892345626347aac00e50))

* add AmazonPolarityClassification dataset ([`75684ed`](https://github.com/embeddings-benchmark/mteb/commit/75684ed8c68bebbfca49a59d943dea3c3e5c791f))

* hack fix bug loading tasks twice ([`23cc372`](https://github.com/embeddings-benchmark/mteb/commit/23cc37245149855565919e576dd9ae22a82cd385))

* add AmazonReviewsClassification ([`5f4731c`](https://github.com/embeddings-benchmark/mteb/commit/5f4731c52c80b38d47ac3a6bd3899ee8aec88b78))

* add create data script for amazon reviews multi ([`761b70e`](https://github.com/embeddings-benchmark/mteb/commit/761b70eadfbf6619986d50b5d3435b0fd6f0d97b))

* make shuffling reproducible in logReg-10-splits-5-intents ([`52e4743`](https://github.com/embeddings-benchmark/mteb/commit/52e47437886bfcd3dc71d8dfb184ca44c7764229))

* add logReg-10-splits-5-intents for kNNClassificationEvaluator ([`72c67e0`](https://github.com/embeddings-benchmark/mteb/commit/72c67e0622b8221c693ad7e001253f52bf91d13b))

* quick fix batch size ([`a88d8bf`](https://github.com/embeddings-benchmark/mteb/commit/a88d8bf3ace3feaa4133a3d5d5357d83ac2d6776))

* quick fixes ([`d4e5549`](https://github.com/embeddings-benchmark/mteb/commit/d4e55499d67a913ba13e46077790aa2f2007da70))

* add batch size to kNNClassificationEvaluator ([`c5127d8`](https://github.com/embeddings-benchmark/mteb/commit/c5127d8adaef7904c8fb8758c39ccae55aa37638))

* Merge pull request #3 from embeddings-benchmark/cross-lingual

Cross lingual ([`c844875`](https://github.com/embeddings-benchmark/mteb/commit/c8448756cd19fbd137c644e0e8bb7f74849840b1))

* black ([`a6ce618`](https://github.com/embeddings-benchmark/mteb/commit/a6ce618e73664461551508d045c1bfe823bba91c))

* bitext mining evaluator ([`db7a934`](https://github.com/embeddings-benchmark/mteb/commit/db7a934a696bfd4dd016bff853aee07d81d99bc1))

* bucc ([`96848c1`](https://github.com/embeddings-benchmark/mteb/commit/96848c182b5230657956cc6b1160f65a2cc488de))

* tatoeba ([`4783ace`](https://github.com/embeddings-benchmark/mteb/commit/4783ace1a5ab717324a861e63ba254668ef979a2))

* bitext mining ([`e0ec3a5`](https://github.com/embeddings-benchmark/mteb/commit/e0ec3a58e99dfecc8526440b37ef04bb9f86a992))

* bitext mining ([`50b2f48`](https://github.com/embeddings-benchmark/mteb/commit/50b2f48e7f511436bd32d51456b8c8388b326b88))

* add MTOP classification tasks ([`1ecec57`](https://github.com/embeddings-benchmark/mteb/commit/1ecec5751b34c7be01b0446a7fcf5ad5d01b7389))

* crosslingual tasks ([`582aa15`](https://github.com/embeddings-benchmark/mteb/commit/582aa15413158b08bd2cdd28dfcfe5937a0f5a68))

* STS17 benchmark ([`0c38bf0`](https://github.com/embeddings-benchmark/mteb/commit/0c38bf037f6e6d5687227c35e16177903aaeab1f))

* add methods ([`49afe21`](https://github.com/embeddings-benchmark/mteb/commit/49afe21a324cc995aa57e04c86df0aefdf3fb3ec))

* formatting ([`cebaf56`](https://github.com/embeddings-benchmark/mteb/commit/cebaf56c8b3a5fd67b8f7d1f4cb4d1878495cde2))

* quick fix ([`2284d17`](https://github.com/embeddings-benchmark/mteb/commit/2284d179612818472f157d7b9769f6e58a64fb94))

* Merge pull request #2 from embeddings-benchmark/knn-classification ([`6a5faec`](https://github.com/embeddings-benchmark/mteb/commit/6a5faeca9e5e2d7edaa9c2ec423a0baea406fd36))

* add MultilingualTask ([`5614a03`](https://github.com/embeddings-benchmark/mteb/commit/5614a031134068194a92f91429bcf7146eb310a2))

* fix loading for multilingual datasets ([`8658f68`](https://github.com/embeddings-benchmark/mteb/commit/8658f6833328cf13c1b88e35f79ea0ceb9398472))

* skip task if results alrdy exist ([`24f83c1`](https://github.com/embeddings-benchmark/mteb/commit/24f83c12910f414bccdd0ec6fee9cbb8ec470a31))

* add banking77 and massive scenario datasets ([`12d4d40`](https://github.com/embeddings-benchmark/mteb/commit/12d4d40deadcb4688957ba7896e78926da057b6b))

* add logRegClassificationEvaluator ([`804e3b0`](https://github.com/embeddings-benchmark/mteb/commit/804e3b0f27c3dabe152ffeacc0b74529fcf504aa))

* add kNNClassificationEvaluatorPytorch ([`31bf4d1`](https://github.com/embeddings-benchmark/mteb/commit/31bf4d1a8670a6959b78f1c94dcd85fc95ceda76))

* cosine and euclidean distances in kNNClassificationEvaluator ([`4720480`](https://github.com/embeddings-benchmark/mteb/commit/4720480b87d40c190764ebd31fab11dbd9992490))

* add requirements dev file ([`a446d3f`](https://github.com/embeddings-benchmark/mteb/commit/a446d3fb81d3e9083a0e95d12efb16360fe5e5a2))

* update results json file format to account for multi langs ([`1fe472a`](https://github.com/embeddings-benchmark/mteb/commit/1fe472a268d712d1a4aa59d393bdebfa7ae6e78f))

* load_dataset directly inside AbsTask ([`4475fee`](https://github.com/embeddings-benchmark/mteb/commit/4475fee551b9e55329c72d9447e06ea5bc03c91c))

* add default language as &#34;en&#34; for all tasks ([`faee9db`](https://github.com/embeddings-benchmark/mteb/commit/faee9db358036eda4aa6810512fb872af16e5d71))

* WIP add kNN Classification and MassiveIntentClassification task ([`885c06d`](https://github.com/embeddings-benchmark/mteb/commit/885c06d0d7d6136e8a0a1b1ddab45f0f1fb6d833))

* tasks can be provided as class now in task_list ([`3bcb767`](https://github.com/embeddings-benchmark/mteb/commit/3bcb76767a8edebe758ff0b13c79a5ee6bcb8ba7))

* add bs param in clusteringevaluator ([`b4c83e0`](https://github.com/embeddings-benchmark/mteb/commit/b4c83e085ad67c9a69d85cdb2a7f16f91073e0a7))

* quick docs fixes ([`1a48f29`](https://github.com/embeddings-benchmark/mteb/commit/1a48f29ccca9427c6d6d154133b172c3c06ad039))

* fix line length ([`faa978f`](https://github.com/embeddings-benchmark/mteb/commit/faa978f9663312d79567d9b15fe7f364b0a762df))

* linting ([`49a4138`](https://github.com/embeddings-benchmark/mteb/commit/49a4138c57d944bdb0c1b4935df008ceee74d449))

* add reqs ([`006756c`](https://github.com/embeddings-benchmark/mteb/commit/006756c94d8ed1e587d506c2be79d34df9413726))

* redditp2p + sep2p ([`2ec9c44`](https://github.com/embeddings-benchmark/mteb/commit/2ec9c4495ec1aebb5588327c44eb022c9e2dca47))

* clustering tasks ([`b8d37a0`](https://github.com/embeddings-benchmark/mteb/commit/b8d37a02c5e3f3196875187c846341259d632801))

* scripts ([`2760969`](https://github.com/embeddings-benchmark/mteb/commit/27609690b59e65eb7b988e039a53c813dbb23afd))

* first commit ([`7fbd064`](https://github.com/embeddings-benchmark/mteb/commit/7fbd064b6848e8dcdad5f4e9ac534a3f0021ff97))

* loading scripts ([`24a4310`](https://github.com/embeddings-benchmark/mteb/commit/24a4310d671f90e3adeb9e000378cb344f6d1723))

* Update README.md ([`c03618c`](https://github.com/embeddings-benchmark/mteb/commit/c03618c9baf60f74e9a022a0792d98b091550c40))

* init file ([`fd182b6`](https://github.com/embeddings-benchmark/mteb/commit/fd182b6ea5d4bb83325891c34e04d0f7bc62c673))

* Update README.md ([`97c6a99`](https://github.com/embeddings-benchmark/mteb/commit/97c6a99ae597b6b1fafbaf442f8f8327b92d608d))

* retrieval evaluator ([`39db013`](https://github.com/embeddings-benchmark/mteb/commit/39db013d20ab9b25c73e24547e22d9ae060f3a19))

* removed results folder ([`751e1fd`](https://github.com/embeddings-benchmark/mteb/commit/751e1fd2f8409fd588e003b4aea8b5ac4a1d8dac))

* reranking evaluator ([`b62a0f5`](https://github.com/embeddings-benchmark/mteb/commit/b62a0f5269f99be48cd52172cdf0fc40aa408c55))

* added custom evaluators ([`1bf7c94`](https://github.com/embeddings-benchmark/mteb/commit/1bf7c949aa7b215cdf97c85660a38838f272142d))

* STS datasets ([`9093bc1`](https://github.com/embeddings-benchmark/mteb/commit/9093bc16589c3355e3493df9f3ded040e4774da3))

* gitignore ([`8d309e4`](https://github.com/embeddings-benchmark/mteb/commit/8d309e456e2db65c7afadadbf4c6fbedb6779b59))

* added STS ([`3a2f4b9`](https://github.com/embeddings-benchmark/mteb/commit/3a2f4b92fec92c42eb6f1d7220c9f26aedc9ed3c))

* reranking ([`0cf6e1a`](https://github.com/embeddings-benchmark/mteb/commit/0cf6e1a25173074d0401fb3c8a269fb64e4be1e1))

* binary classification ([`3a15b96`](https://github.com/embeddings-benchmark/mteb/commit/3a15b96b9bf5eaefa3b8ccd6f0c2e333b94620b0))

* added verbosity level ([`1ee8f3d`](https://github.com/embeddings-benchmark/mteb/commit/1ee8f3d2d48bb2f8b980131850cd3b98e54aad98))

* added file logging ([`36f7cf3`](https://github.com/embeddings-benchmark/mteb/commit/36f7cf3d2fce8d565bf75ed76df4727f6748ede6))

* added available tasks/categories/selected list ([`5cc63a5`](https://github.com/embeddings-benchmark/mteb/commit/5cc63a5e4ee29d74ba217f48dec2c6785ca71355))

* finegrained task selection ([`7c0087b`](https://github.com/embeddings-benchmark/mteb/commit/7c0087b3561297e77d58bbd61955dfbc89c61450))

* added retrieval ([`22415e9`](https://github.com/embeddings-benchmark/mteb/commit/22415e9d12400d900337e45aefc55942ca537ae8))

* fixed seed ([`50ada77`](https://github.com/embeddings-benchmark/mteb/commit/50ada7738f0062185b289a31c765871fb2a3e2fc))

* typos ([`16dc4a9`](https://github.com/embeddings-benchmark/mteb/commit/16dc4a98c1d5a55be997ed319d9a34e272949583))

* added clustering tasks ([`1106c15`](https://github.com/embeddings-benchmark/mteb/commit/1106c1565c1d5cb7fdb439ee6ade6b70bbc6fea9))

* seeded benchmarks ([`518bc82`](https://github.com/embeddings-benchmark/mteb/commit/518bc8248381d38e7b19fbb434f86c7d417a839a))

* evaluation schema ([`bdb79d0`](https://github.com/embeddings-benchmark/mteb/commit/bdb79d0f4fd2635cc54072831a3615bc7b8eb74b))

* basic tasks schema ([`bacb9d0`](https://github.com/embeddings-benchmark/mteb/commit/bacb9d078fa7c7fa10d8b5e3fdf2a41c2beaf97e))

* proof of concept ([`6886d1b`](https://github.com/embeddings-benchmark/mteb/commit/6886d1b50f0c16ed1b86b5baec9d3853f02f9256))

* Create README.md ([`26df27b`](https://github.com/embeddings-benchmark/mteb/commit/26df27b61071ea8b6000f771c4a68412d8b2d223))

* Initial commit ([`7841bca`](https://github.com/embeddings-benchmark/mteb/commit/7841bca7daeea0473b4cfdfe37f0a290feba6b8f))
