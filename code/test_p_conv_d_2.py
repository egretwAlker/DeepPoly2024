import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'conv_d', '--spec', 'preliminary_test_cases/conv_d/img10_mnist_0.084488.txt']
main()