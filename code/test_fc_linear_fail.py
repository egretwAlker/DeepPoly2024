import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'fc_linear', '--spec', 'test_cases/fc_linear/img_mnist_0.077340.txt']
main()