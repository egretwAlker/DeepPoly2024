import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'fc6_dw', '--spec', 'test_cases/fc6_dw/img_mnist_0.034540.txt']
main()