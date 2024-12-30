import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'fc6_base', '--spec', 'test_cases/fc6_base/img_mnist_0.074530.txt']
main()