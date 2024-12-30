import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'fc_base', '--spec', 'my_test_cases/fc_base/img_mnist_0.047839.txt']
main()