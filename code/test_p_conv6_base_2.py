import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'conv6_base', '--spec', 'preliminary_test_cases/conv6_base/img10_cifar10_0.003959.txt']
main()