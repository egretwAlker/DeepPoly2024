import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'fc6_w', '--spec', 'test_cases/fc6_w/img_mnist_0.042779.txt']
main()