import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'skip6_large', '--spec', 'test_cases/skip6_large/img_mnist_0.019448.txt']
main()