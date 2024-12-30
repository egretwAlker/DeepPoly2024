import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'skip', '--spec', 'test_cases/skip/img_mnist_0.078863.txt']
main()