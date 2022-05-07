#!/bin/sh
black . --extend-exclude .ipynb -v --exclude="/build|dist|benchmarks|boyd\-admm/"
flake8 . --count --ignore="E501 W503 E203 F841" --show-source --statistics --exclude=./.*,benchmarks,build,dist,boyd-admm
