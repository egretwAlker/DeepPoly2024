import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'fc_dw', '--spec', 'test_cases/fc_dw/img_mnist_0.030104.txt']
main()