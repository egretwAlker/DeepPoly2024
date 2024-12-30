import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'conv_linear', '--spec', 'test_cases/conv_linear/img_mnist_0.072113.txt']
main()