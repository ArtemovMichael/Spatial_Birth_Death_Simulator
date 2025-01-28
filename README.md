# Spatial Birth-Death Simulator (Cython Wrapper)

Directory structure for this project looks like:

```
SBDPP_sim/
├── examples/
├── include/
│   └── SpatialBirthDeath.h
├── src/
│   └── SpatialBirthDeath.cpp
├── simulation/
│   ├── __init__.py
│   └── SpatialBirthDeathWrapper.pyx
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```
