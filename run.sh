#!/bin/bash

python run.py --fn sin_euclid_lo --epochs 1000
python run.py --fn sin_euclid_mi --epochs 1000
python run.py --fn sin_euclid_hi --epochs 1000
python run.py --fn sin_euclid_lo_biased --epochs 1000
python run.py --fn sin_euclid_mi_biased --epochs 1000
python run.py --fn sin_euclid_hi_biased --epochs 1000
python run.py --fn std_normal --epochs 1000
python run.py --fn half_unit_sphere --epochs 1000
python run.py --fn sinc --epochs 1000
python run.py --fn sc_lo --epochs 1000
python run.py --fn sc_mi --epochs 1000
python run.py --fn sc_hi --epochs 1000
python run.py --fn block --epochs 1000

python run.py --fn sin_euclid_lo --epochs 1000 --pe 1
python run.py --fn sin_euclid_mi --epochs 1000 --pe 1
python run.py --fn sin_euclid_hi --epochs 1000 --pe 1
python run.py --fn sin_euclid_lo_biased --epochs 1000 --pe 1
python run.py --fn sin_euclid_mi_biased --epochs 1000 --pe 1
python run.py --fn sin_euclid_hi_biased --epochs 1000 --pe 1
python run.py --fn std_normal --epochs 1000 --pe 1
python run.py --fn half_unit_sphere --epochs 1000 --pe 1
python run.py --fn sinc --epochs 1000 --pe 1
python run.py --fn sc_lo --epochs 1000 --pe 1
python run.py --fn sc_mi --epochs 1000 --pe 1
python run.py --fn sc_hi --epochs 1000 --pe 1
python run.py --fn block --epochs 1000 --pe 1
