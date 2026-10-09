PY ?= python3
# Private paths: set RRC_ANNOTATION_ROOT / RRC_DATA_DIR, or copy configs/local.example.json to configs/local.json

.PHONY: test data s1 s2 softmax

test:
	$(PY) -m pytest -q tests

# data stage against the checked-in manifests (freeze-grid / split are one-time and refuse to overwrite)
data:
	$(PY) -m rrc audit
	$(PY) -m rrc dedup
	$(PY) -m rrc check
	$(PY) -m rrc datasets
	$(PY) -m rrc report

s1:
	$(PY) -m rrc s1-b0
	$(PY) -m rrc s1-neural --model m1_arabert
	$(PY) -m rrc s1-neural --model m1_arabertv02
	$(PY) -m rrc s1-neural --model m2_camelbert
	$(PY) -m rrc s1-aggregate

s2:
	$(PY) -m rrc s2-baselines
	$(PY) -m rrc s2-neural --model m1_arabert
	$(PY) -m rrc s2-neural --model m1_arabertv02
	$(PY) -m rrc s2-neural --model m2_camelbert
	$(PY) -m rrc s2-aggregate

softmax:
	$(PY) -m rrc s1-softmax --model m1s_arabert
	$(PY) -m rrc s1-softmax --model m1s_arabertv02
	$(PY) -m rrc s1-softmax --model m2s_camelbert
