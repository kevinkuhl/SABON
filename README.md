# SABON
Official GitHub repository for **S**ingle **A**utoencoder **B**asis **O**perator **N**etworks  
<https://arxiv.org/abs/2505.05085>

---

## 1 - What is SABON?
A *single autoencoder basis operator network* jointly learns  

* a **basis** $\phi$, and  
* the **finite matrix representation** of the PF/Koopman operators in that same basis.

---

## 2 - Directory layout
```text
├── sabon/                  # core library (model + utils)
├── examples/
│   ├── circle_rotation/    # rotation on S¹
│   └── cat_map/            # Anosov map on T²
└── requirements.txt
````

---

## 3 - Quick-start

### 3.1  Install

```bash
conda create -n sabon python=3.11
conda activate sabon
pip install -r requirements.txt
```

### 3.2  Circle-rotation demo

```bash
cd examples/circle_rotation

# 1) generate data
python data.py \
  --n_functions 1000 --n_points 100 --max_order 9 --alpha 1 \
  --saving_directory ./data

# 2) train
python train.py \
  --config ./config.yaml \
  --save_dir ./checkpoints

# 3) analyse results
python analysis.py \
  --checkpoints_dir ./checkpoints \
  --output_dir ./results
```

### 3.3  Perturbed cat-map demo

```bash
cd examples/cat_map

python data.py \
  --n_functions 4000 --n_points 100 --max_order 5 \
  --saving_directory ./data

python train.py \
  --config ./config.yaml \
  --save_dir ./checkpoints

python analysis.py \
  --checkpoint_dir ./checkpoints \
  --output_dir   ./results
```
