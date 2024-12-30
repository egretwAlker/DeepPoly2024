import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'conv_base', '--spec', 'test_cases/conv_base/img_mnist_0.055929.txt']
main()