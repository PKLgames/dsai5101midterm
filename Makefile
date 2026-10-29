# Cat Image Arena — CPU pipeline
#
#   make all        env + images + YOLO/CLIP labels + ImageNet features + models
#   make help       list targets
#
# Override workers/threads:  make all WORKERS=2 THREADS=4
# Force a YOLO re-label:     make labels

SHELL := /bin/bash
.ONESHELL:
.DELETE_ON_ERROR:
.DEFAULT_GOAL := help

PYTHON            ?= .venv/bin/python
# Touched after a requirements-matching install. Artifact rules list it as an
# order-only prerequisite so recreating the venv does not invalidate caches.
ENV_STAMP         := .venv/.setup-stamp
WORKERS           ?= 4
THREADS           ?= 8
DOWNLOAD_WORKERS  ?= 16
TAG               ?= mobilenet_v3_small_224

MANIFEST          := artifacts/manifest.csv
INDEX             := artifacts/image_index.csv
PHOTO_LABELS      := artifacts/labels/photo_labels.csv
FEATURES_MNV3     := artifacts/features/$(TAG).npz
FEATURES_RESNET   := artifacts/features/resnet18_224.npz
QA_SUMMARY        := artifacts/features/qa_summary_$(TAG).json
BASELINE          := artifacts/submissions/pred_test.csv
FAMILIES          := artifacts/features/feature_families.json
COLD_START        := artifacts/features/cold_start_fix.json
ROUTED            := artifacts/submissions/pred_routed_test.csv
JOINT             := artifacts/submissions/pred_hybrid_test.csv
TABLES            := tables/train_pairs.csv.gz

# $(call BANNER,one-sentence description) — no commas in the text (Make splits on them).
BANNER = printf '\n%s\n%s\n%s\n' \
	'----------------------------------------------------------------' \
	'==> $(1)' \
	'----------------------------------------------------------------'

.PHONY: help all env images index labels features qa models tables \
        baseline families coldstart hybrid joint quickstart \
        clean clean-features clean-models distclean

help:
	@echo "Cat Image Arena — CPU pipeline"
	@echo
	@echo "  make all         env + download + YOLO/CLIP labels + ImageNet features + models"
	@echo "  make env         CPU-only .venv (Python 3.14; macOS arm64 / Linux x86_64+arm64)"
	@echo "  make images      1,558 photos from Hugging Face (~184 MB)"
	@echo "  make labels      YOLO11n-seg + CLIP photo labels (slow on CPU; --fresh)"
	@echo "  make features    frozen ImageNet embeddings (MobileNetV3-Small + ResNet18)"
	@echo "  make qa          same-cat vs different-cat diagnostics"
	@echo "  make models      baseline + feature families + cold-start + routed hybrid"
	@echo "  make tables      ready-to-use train/test csv.gz"
	@echo "  make quickstart  embed one photo and check it against the cache"
	@echo "  make clean       features, submissions, tables, index, manifest"
	@echo "  make distclean   that plus .venv, images, labels, torch/uv caches, YOLO weights"
	@echo
	@echo "'make all' runs YOLO/CLIP if $(PHOTO_LABELS) is missing."
	@echo "Run 'make labels' to relabel from scratch."
	@echo "WORKERS=$(WORKERS)  THREADS=$(THREADS)  DOWNLOAD_WORKERS=$(DOWNLOAD_WORKERS)"

# --------------------------------------------------------------------------- #
# Full analysis
# --------------------------------------------------------------------------- #

all: env images index $(PHOTO_LABELS) features qa models tables
	@$(call BANNER,Done. Submissions are in artifacts/submissions/ and tables in tables/.)

env: $(ENV_STAMP)

images: $(MANIFEST)

index: $(INDEX)

features: $(FEATURES_MNV3) $(FEATURES_RESNET)

qa: $(QA_SUMMARY)

models: baseline families coldstart hybrid

tables: $(TABLES)

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

# Recreate .venv only when requirements.txt or the setup script changed.
# A newer checkout mtime, or a rebuilt interpreter, must not miss the feature,
# label, or model caches: those rules depend on this stamp only as order-only.
$(ENV_STAMP): requirements.txt scripts/00_setup_env.sh
	@$(call BANNER,Build a CPU-only Python 3.14 virtualenv and install pinned dependencies.)
	bash scripts/00_setup_env.sh
	touch $@

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

$(MANIFEST): course/train.csv course/test.csv | $(ENV_STAMP)
	@$(call BANNER,Download the 1558 cat photos from Hugging Face.)
	$(PYTHON) scripts/01_download_images.py --workers $(DOWNLOAD_WORKERS)

$(INDEX): $(MANIFEST) | $(ENV_STAMP)
	@$(call BANNER,Build image_index.csv and check train/test overlap.)
	$(PYTHON) scripts/02_build_index.py

# YOLO + CLIP. Order-only prereqs: a newer venv or download must not force a
# relabel. `make labels` always re-runs with --fresh.
$(PHOTO_LABELS): | $(ENV_STAMP) $(MANIFEST)
	@$(call BANNER,Label every photo with YOLO11n-seg masks and CLIP attributes.)
	$(PYTHON) photo-labelling/label_photos.py

labels: | $(ENV_STAMP) $(MANIFEST)
	@$(call BANNER,Relabel every photo from scratch with YOLO11n-seg and CLIP.)
	$(PYTHON) photo-labelling/label_photos.py --fresh

# --------------------------------------------------------------------------- #
# Frozen ImageNet features
# --------------------------------------------------------------------------- #

$(FEATURES_MNV3): $(INDEX) | $(ENV_STAMP)
	@$(call BANNER,Extract frozen MobileNetV3-Small ImageNet embeddings for every photo.)
	$(PYTHON) scripts/03_extract_features.py \
		--backbone mobilenet_v3_small --workers $(WORKERS) --threads $(THREADS)

$(FEATURES_RESNET): $(INDEX) | $(ENV_STAMP)
	@$(call BANNER,Extract frozen ResNet18 ImageNet embeddings for every photo.)
	$(PYTHON) scripts/03_extract_features.py \
		--backbone resnet18 --workers $(WORKERS) --threads $(THREADS)

$(QA_SUMMARY): $(FEATURES_MNV3) | $(ENV_STAMP)
	@$(call BANNER,Check that embeddings separate same-cat photos from different-cat photos.)
	$(PYTHON) scripts/04_feature_qa.py --tag $(TAG)

# --------------------------------------------------------------------------- #
# Preference models
# --------------------------------------------------------------------------- #

baseline: $(BASELINE)

$(BASELINE): $(FEATURES_MNV3) | $(ENV_STAMP)
	@$(call BANNER,Fit the embedding preference baseline and write pred_test.csv.)
	$(PYTHON) scripts/05_baseline_pairs.py --tag $(TAG)

families: $(FAMILIES)

$(FAMILIES): $(FEATURES_MNV3) $(PHOTO_LABELS) | $(ENV_STAMP)
	@$(call BANNER,Compare feature families for predicting which photo an observer chooses.)
	$(PYTHON) scripts/07_feature_families.py --tag $(TAG)

coldstart: $(COLD_START)

$(COLD_START): $(FEATURES_MNV3) $(FEATURES_RESNET) $(PHOTO_LABELS) | $(ENV_STAMP)
	@$(call BANNER,Test routing versus extra features on unseen photos and unseen cats.)
	$(PYTHON) scripts/08_cold_start_fix.py

hybrid: $(ROUTED)

$(ROUTED): $(FEATURES_MNV3) $(PHOTO_LABELS) | $(ENV_STAMP)
	@$(call BANNER,Fit the routed hybrid model and write the test submission.)
	$(PYTHON) scripts/09_hybrid_submission.py --structure routed --tag $(TAG)

joint: $(JOINT)

$(JOINT): $(FEATURES_MNV3) $(PHOTO_LABELS) | $(ENV_STAMP)
	@$(call BANNER,Fit the joint hybrid model and write pred_hybrid_test.csv.)
	$(PYTHON) scripts/09_hybrid_submission.py --structure joint --tag $(TAG)

$(TABLES): $(FEATURES_MNV3) | $(ENV_STAMP)
	@$(call BANNER,Export joined train/test tables for modelling without this repo.)
	$(PYTHON) scripts/06_export_tables.py --tag $(TAG)

quickstart: $(FEATURES_MNV3) | $(ENV_STAMP) $(MANIFEST)
	@$(call BANNER,Embed one photo and check it against the feature cache.)
	$(PYTHON) scripts/10_quickstart.py --image artifacts/images/cat_000002/cat_000002_0001.jpg

# --------------------------------------------------------------------------- #
# Clean
# --------------------------------------------------------------------------- #
# clean      — derived outputs (features, submissions, tables, index)
# distclean  — that plus .venv, images, labels, package/weight caches
# After distclean, `make all` re-runs YOLO/CLIP because photo_labels.csv is gone.

clean: clean-features clean-models
	rm -f $(INDEX) $(MANIFEST) artifacts/download_errors.csv
	rm -f $(TABLES) tables/test_pairs.csv.gz tables/image_features.csv.gz

clean-features:
	rm -rf artifacts/features

clean-models:
	rm -rf artifacts/submissions

distclean: clean
	rm -rf .venv .uv-cache .cache models
	rm -rf artifacts/images artifacts/captioned-photos artifacts/labels
	find . -type d -name __pycache__ -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true