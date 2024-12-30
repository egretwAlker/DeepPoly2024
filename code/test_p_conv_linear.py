import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'conv_linear', '--spec', 'preliminary_test_cases/conv_linear/img6_mnist_0.052523.txt']
main()