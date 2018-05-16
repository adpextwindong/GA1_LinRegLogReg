# Group Assignment 3
George Crary, David Barnes, Griffin Gonsalves

## Invocations
Install imports, --user may be needed. This runs on the CS434 VM.
`pip install -r requirements.txt`

Running the Cifar solution indivisually
`python cifar_p3.py 1 SIG 10 0.001 0.1 0.5 0.25 > 1_SIG_10_0.001_0.1_0.5_0.25_out.txt`

Running the tests
`./test.sh`

Producing pyplot report. Requires out.txt from test.sh runs.
`python reporting.py; open *.png`

Doing the whole thing
`./test.sh; python reporting.py; open *.png`

### Command line arguments
See docopt string at the top (or just run it without args) to see the commands.

Usage:
        cifar_p3.py <depth> <activation> <epochs> <learning_rate> <dropout> <momentum> <weight_decay>

    Arguments:
        depth: INT - 1 as Single or a value > 1 for multilayer
        activation: SIG - Sigmoid Function
            RELU - ReLu Activation Function
        epochs: INTEGER
        learning_rate: FLOAT
        dropout: FLOAT
        momentum: FLOAT
        weight_decay: FLOAT


