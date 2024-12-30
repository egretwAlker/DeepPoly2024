import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'fc_base', '--spec', 'test_cases/fc_base/img_mnist_0.048839.txt']
main()