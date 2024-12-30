import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'fc6_d', '--spec', 'test_cases/fc6_d/img_mnist_0.089335.txt']
main()