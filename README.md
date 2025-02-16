# Homework 2 Programming - Gaussian Mixture Models

In this homework, you will implement the expectation-maximization algorithm for a Gaussian mixture model.

## Instructions

To complete the assignment, you must implement the following functions:

1. `initialize_parameters`
2. `compute_sigma`
3. `prob`
4. `E_step`
5. `M_step`
6. `likelihood`
7. `train`

## Installing Required Packages

You will need to install the following packages:

```bash
pip install numpy matplotlib scipy pytest
```

## Testing your implementation

You can test your implementation by running the following command:

```bash
python test_gmm.py
```

For specific tests, you can run:

```bash
pytest test_gmm.py -k test_gmm_initialization
```

## Submission

Submit `submssion.py` file with the implemented functions on Gradescope.
