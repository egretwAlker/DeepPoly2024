import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'skip6', '--spec', 'test_cases/skip6/img_mnist_0.102399.txt']
main()