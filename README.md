# Provenance-aware UI structure recovery

COMP5064 capstone project. Given a screenshot of an Android app screen or a web page, the boxes of its visible
elements, and whatever structural metadata came with it (an Android view hierarchy or a web page tree, which may be
correct, wrong, partly missing or absent), the system recovers the screen's element tree: the type of every element
and how elements nest inside containers. It also gives each element a trust score and marks, for every container,
whether it was read from the metadata or inferred by the model.

Data: RICO screenshots and view hierarchies with CLAY's cleaned labels (Android), and WebUI pages (web). Everything
is written in Python with PyTorch.

Contents: [the models](#the-models) · [folders](#folders) · [file guide](#file-guide) ·
[where each report result comes from](#where-each-report-result-comes-from) · [setup](#setup) · [data](#data) ·
[reproducing the results](#reproducing-the-results) · [Colab notes](#colab-notes)

## The models

| name in the report | what it is | code |
|---|---|---|
| Model 1 | MLP that predicts each element's type from metadata only (box, raw-tree position, flags, class name) | `src/uistruct/models/type_mlp.py` |
| Model 2 | Model 1 + 18 colour statistics from the element's image crop | `src/uistruct/data/colour.py` |
| Model 3 | Model 1 + a fine-tuned ResNet-18 embedding of the element's crop | `src/uistruct/models/cnn.py` |
| Models 4, 5, 6 | one transformer scorer that picks containers from candidates: metadata + pixels (4), pixels only (5), metadata only (6) | `src/uistruct/models/fusion.py`, `src/uistruct/models/nesting.py` |
| fusion network + trust (final Android system) | Model 4's network with a per-element trust head, trained on correct, corrupted and missing metadata | `src/uistruct/models/fusion.py` (`trust=True`), `src/uistruct/train/provenance_head.py` |
| joint network / web-trained network | the same network trained on Android and web pages together, or on web pages only | `scripts/train_web_only_and_joint_android_web_networks.py` |

## Folders

| folder | contents |
|---|---|
| `src/uistruct/` | the Python package: data loading and features (`data/`), models (`models/`), training (`train/`), metrics and statistics (`eval/`), run logging, demo pipeline |
| `scripts/` | command-line entry points, one per experiment; each reads one config |
| `configs/` | one YAML file per experiment, named like the script that reads it; every setting lives here |
| `notebooks/` | Colab notebooks for the GPU runs, saved with their outputs |
| `tests/` | pytest suite |
| `results/` | the record of every run (see below) and the data-analysis files |

Every run writes `results/runs/<name>_run<N>/` with `config.yaml` (the exact settings), `env.json` (Python, package
and GPU versions, seed, and the code version it ran from) and `metrics.json` (all scores, with 95% confidence
intervals), plus per-screen score tables where useful. Runs made on Colab are grouped in `results/runs/colab_*/` and
`results/runs/web_train_colab/`.

## File guide

### Package `src/uistruct/`

| file | what it does |
|---|---|
| `runlog.py` | creates each run folder and writes its config, environment and metrics |
| `plotting.py` | shared colours and style for the figures |
| `demo.py` | the end-to-end system for one screen: metadata at a chosen condition in, labelled tree with trust scores and provenance out |
| `data/rico_clay.py` | joins RICO view hierarchies with CLAY labels into one flat node table |
| `data/splits.py` | leakage-free splits: no company and no identical layout shared between training, validation and test |
| `data/tree.py` | the true elements and true tree of each screen from CLAY's labels |
| `data/features.py` | Model 1 inputs: box, raw-tree position, flags, class name and its pieces |
| `data/colour.py` | Model 2's 18 colour statistics per element crop |
| `data/crops.py` | image crops for the CNN (box + 8 px margin, screenshot resized to 540 x 960) |
| `data/nesting_tokens.py` | per-screen inputs of Models 4-6: leaves and candidate containers with their relations |
| `data/nesting_crops.py` | crop embeddings for the leaves and candidates of Models 4 and 5 |
| `data/provenance_tokens.py` | per-screen inputs of the trust network, the metadata corruption levels and the trust labels |
| `data/webui.py` | WebUI pages: download of only the needed files, page tree to the shared element format, true containers |
| `models/type_mlp.py` | Model 1 network |
| `models/cnn.py` | Model 3 network (ResNet-18 crop encoder + Model 1 MLP) |
| `models/baselines.py` | non-neural type baselines (majority class, class-name lookup) |
| `models/nesting.py` | candidate generation (XY-cut, agglomerative clustering), the assembler that builds a valid tree, the oracle for the pool ceiling |
| `models/nesting_raw.py` | helpers for the raw view hierarchy (for example collapsing single-child wrappers) |
| `models/fusion.py` | the relation-aware transformer scorer of Models 4-6 and the trust head |
| `train/type_head.py` | training and scoring of Models 1 and 2 |
| `train/cnn_head.py` | training and scoring of Model 3 (frozen or fine-tuned backbone) |
| `train/nesting_head.py` | training and scoring of Models 4-6 |
| `train/provenance_head.py` | training and scoring of the trust network on corrupted metadata |
| `eval/metrics.py` | every metric in the report: type macro-F1, container match, container precision and F1 |
| `eval/stats.py` | bootstrap confidence intervals over screens and paired bootstrap tests |
| `eval/trust.py` | AUROC, average precision, Brier skill score, calibration error and reliability bins of the trust score |
| `eval/type_report.py` | shared type-head report for several models scored on the same rows |

### Scripts `scripts/`

| script | what it does |
|---|---|
| `download_data.sh` | downloads CLAY (pinned commit) and RICO, then runs `check_raw_data.py` |
| `check_raw_data.py` | checks the raw download counts |
| `build_node_table.py` | builds the joined node table |
| `build_splits_and_targets.py` | builds the leakage-free splits and the true trees |
| `explore_data.py` | data analysis: class balance, metadata noise, screen sizes |
| `plot_box_overlay.py` | visual check that boxes line up with screenshots |
| `structure_rule_baselines_true_elements.py` | first rule baselines; also exercises the whole evaluation code |
| `measure_structure_inputs.py` | measures what the structure step receives (containers, single-child containers) |
| `train_element_types_metadata_mlp.py` | Model 1: tuning, validation and test, with its baselines and seeds |
| `plot_element_types_metadata_mlp.py` | Model 1 figure |
| `extract_colour_features.py` | computes Model 2's colour features once for every element |
| `train_element_types_with_colour.py` | Model 2: tuning, validation and test |
| `plot_element_types_with_colour.py` | Model 2 figure |
| `colab_gpu_speed_check.py` | GPU speed test used to plan the Colab runs |
| `pack_colab_data_element_types.py` | packs the screenshots and tables Model 3 needs on Colab |
| `train_element_types_cnn_crops.py` | Model 3: crop cache, frozen and fine-tuned backbone, tuning, validation and test |
| `structure_rule_baselines.py` | rule baselines for every metadata condition (XY-cut, raw hierarchy, collapsed raw hierarchy) |
| `structure_candidate_pool_ceiling.py` | best score reachable from each candidate pool |
| `pack_colab_data_structure.py` | packs the tables Models 4-6 need on Colab |
| `train_structure_models.py` | Models 4-6 and the gradient-boosted trees on the same candidates |
| `analyse_saved_outputs.py` | re-scores saved outputs (screens with no true container; leaf types of Models 1-6) |
| `train_trust_score_on_corrupted_metadata.py` | the trust network: error rates, corruption, training, thresholds, test |
| `analyse_trust_score_results.py` | analysis and figures of the trust network runs |
| `evaluate_android_system_on_web_pages.py` | the Android-trained system on WebUI pages |
| `train_web_only_and_joint_android_web_networks.py` | web-only and joint Android + web networks |
| `demo_checks_and_screen_choice.py` | checks the demo against the batch test results and picks the demo screens by a fixed rule |
| `demo_app.py`, `demo_page.html` | the local browser application |
| `plot_system_overview.py` | system diagram |

### Notebooks `notebooks/` (Colab GPU)

| notebook | runs |
|---|---|
| `colab_gpu_speed_check.ipynb` | GPU speed test |
| `train_element_type_classifier_with_resnet18_crops.ipynb` | Model 3 |
| `train_structure_models_pixels_metadata_fusion.ipynb` | Models 4-6 |
| `train_trust_score_on_corrupted_metadata.ipynb` | the trust network and its variants |
| `train_web_only_and_joint_android_web_networks.ipynb` | web-only and joint networks |

Execution counts show the order in which the cells were really run; some cells were run again later in the same
session.

### Tests `tests/`

`test_rico_clay.py` (node table), `test_eval.py` (metrics and bootstrap), `test_element_types_mlp.py` (Model 1),
`test_colour_features.py` (Model 2 features), `test_resnet18_crops.py` (Model 3 crops), `test_structure_scorer.py`
(candidates and assembler), `test_fusion_scorer.py` (transformer scorer), `test_structure_models.py` (Models 4-6
inputs, tuning draws, trees comparator), `test_trust_score_inputs.py` (corruption and trust labels), `test_demo.py` (demo pipeline).

## Where each report result comes from

| report item | file with the numbers |
|---|---|
| Section 2 data numbers, Figure 1, Figure A4 | `results/explore_data.json`, `results/node_table_check.json` |
| Splits, Table A5 | `results/splits_and_targets.json` |
| Figure 3a (types, Models 1-3) | `results/runs/colab_element_types_cnn/element_types_cnn_test_run1/metrics.json` |
| Model 1 and Model 2 test details | `results/runs/element_types_mlp_test_run1/`, `results/runs/element_types_colour_test_run1/` |
| Figure 3b, Table A2 (structure, Models 4-6) | `results/runs/saved_outputs_rescore_run1/metrics.json`; three seeds: `results/runs/colab_trust_score/structure_leftover_test_run1/metrics.json` |
| Table A6 (leaf types) | `results/runs/saved_outputs_leaf_types_run2/metrics.json` |
| Figure 4, Table A3 (five metadata conditions) | `results/runs/colab_trust_score/trust_score_test_run1/metrics.json` (`structure`, `paired`, `seeds`) |
| Figure 5, Table A4 (trust) | the same file (`trust`) |
| type-head removal and comparison with Model 5 | `results/runs/trust_score_analysis_run1/metrics.json` |
| Figure 6, Table A7 (web) | `results/runs/web_train_colab/runs/web_train_test_run1/metrics.json` |
| Android-trained system on the web, first test | `results/runs/web_test_run1/metrics.json` |
| Figure A1, 999 of 1,000 demo runs match | `results/runs/demo_check_run1/metrics.json` |
| Figure A2 screens | `results/runs/demo_pick_run1/metrics.json` |
| Figure A3 (training curve) | `results/runs/colab_trust_score/trust_score_train_fusion_full_s0_run1/metrics.json` (`history`) |
| Table A1 tuning | `results/runs/element_types_mlp_tune_run1/`, `results/runs/element_types_colour_tune_run1/`, `results/runs/colab_element_types_cnn/element_types_cnn_tune_run2/`, `results/runs/structure_tune_model4_run1/` (and `model5`, `model6`) |

## Setup

Python 3.10 on Windows or Linux (laptop runs used CPU only).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
python -m pytest
```

Some tests need the raw data or trained model files and are skipped when these are missing.

## Data

```bash
bash scripts/download_data.sh        # CLAY (pinned commit) + RICO unique UIs (~6.5 GB), then checks the counts
python scripts/build_node_table.py --config configs/build_node_table.yaml       # one node table, results/node_table_check.json
python scripts/build_splits_and_targets.py --config configs/build_splits_and_targets.yaml  # app-disjoint splits + ground-truth trees
```

`data/` is not in the repository. The WebUI pages are fetched by the `fetch` stages of `scripts/evaluate_android_system_on_web_pages.py`
and `scripts/train_web_only_and_joint_android_web_networks.py`, which download only the needed files from the Hugging Face archives.

## Reproducing the results

Run from the repository root, in this order. Each command writes a new run folder under `results/runs/`.

| step (report name) | command | where |
|---|---|---|
| data analysis | `python scripts/explore_data.py --config configs/explore_data.yaml` | laptop |
| nesting baselines on the true elements | `python scripts/structure_rule_baselines_true_elements.py --config configs/structure_rule_baselines_true_elements.yaml` | laptop |
| what the nesting step receives | `python scripts/measure_structure_inputs.py --config configs/measure_structure_inputs.yaml` | laptop |
| element types from metadata (Model 1) | `python scripts/train_element_types_metadata_mlp.py --config configs/train_element_types_metadata_mlp.yaml --stage tune`, then `dev`, then `test`; figure: `python scripts/plot_element_types_metadata_mlp.py` | laptop |
| colour features | `python scripts/extract_colour_features.py --config configs/colour_features.yaml` | laptop |
| element types with colour (Model 2) | `python scripts/train_element_types_with_colour.py --config configs/train_element_types_with_colour.yaml --stage tune`, then `dev`, then `test`; figure: `python scripts/plot_element_types_with_colour.py` | laptop |
| GPU speed test | `notebooks/colab_gpu_speed_check.ipynb` (`scripts/colab_gpu_speed_check.py`) | Colab |
| element types with ResNet-18 crops (Model 3) | `python scripts/pack_colab_data_element_types.py --tables` and `--extra` (files for Drive), then `notebooks/train_element_type_classifier_with_resnet18_crops.ipynb` (`scripts/train_element_types_cnn_crops.py`, stages features, frozen, cache, tune, dev, test) | laptop, then Colab |
| nesting baselines, all metadata conditions | `python scripts/structure_rule_baselines.py --config configs/structure_rule_baselines.yaml` (validation), then `configs/structure_rule_baselines_test.yaml` (test, once) | laptop |
| candidate-pool ceiling | `python scripts/structure_candidate_pool_ceiling.py --config configs/structure_candidate_pool_ceiling.yaml` | laptop |
| structure models (Models 4-6) | `python scripts/pack_colab_data_structure.py`, then `notebooks/train_structure_models_pixels_metadata_fusion.ipynb` (`scripts/train_structure_models.py --config configs/train_structure_models.yaml`) | laptop, then Colab |
| analysis of saved outputs | `python scripts/analyse_saved_outputs.py rescore`, then `leaf_types` | laptop |
| trust score on corrupted metadata (fusion network + trust) | `python scripts/train_trust_score_on_corrupted_metadata.py --config configs/train_trust_score_on_corrupted_metadata.yaml --laptop --stage rates`, then `fingerprint`, `build --split dev`, `baselines`; then `notebooks/train_trust_score_on_corrupted_metadata.ipynb` (build, embed, train, dev, test) | laptop, then Colab |
| trust analysis and figures | `python scripts/analyse_trust_score_results.py` | laptop |
| demo | `python scripts/demo_checks_and_screen_choice.py --stage check`, `--stage pick`; app: `python scripts/demo_app.py` (needs the trained model files in `data/models/`) | laptop |
| Android-to-web test | `python scripts/evaluate_android_system_on_web_pages.py --config configs/evaluate_android_system_on_web_pages.yaml --stage fetch`, then `build` and `run` for `--split val` and `--split test` | laptop |
| web-trained networks | `notebooks/train_web_only_and_joint_android_web_networks.ipynb` (`scripts/train_web_only_and_joint_android_web_networks.py --config configs/train_web_only_and_joint_android_web_networks.yaml`) | Colab |

Test splits are scored once, after every choice was fixed on validation; the nesting, trust and web scripts
ask for `--confirm-test` before a test run, and the notebooks have a `RUN_TEST` switch. Seeds are fixed in the configs. Bootstrap confidence intervals
resample whole screens (1,000 resamples, seed 0).

## Colab notes

The notebooks run on Google Colab (the first markdown cell of each says which GPU and which files it needs).
Each notebook mounts Google Drive and expects a folder `My Drive/uistruct` holding the small upload files made on
the laptop (`pack_colab_data_*.py`); checkpoints, run folders and reports are written back there, so a disconnected
session resumes where it stopped. The notebooks clone this repository at the start (they ask for a read-only GitHub token, used once and never
saved). Screenshots are downloaded on
Colab from the official RICO server and checked byte for byte against the laptop copy. Colab's own torch build is
used; the versions of every run are in its `env.json`.

## What is not in the repository

Raw and processed data, model checkpoints and the large per-element probability files (`*_probs.npz`) are left out
for size; the scripts rebuild them.
