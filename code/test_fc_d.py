import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'fc_d', '--spec', 'test_cases/fc_d/img_mnist_0.040164.txt']
main()