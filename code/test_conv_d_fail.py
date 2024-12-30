import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'conv_d', '--spec', 'test_cases/conv_d/img_mnist_0.103841.txt']
main()