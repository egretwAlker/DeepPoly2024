import sys
from verifier import main
sys.argv = ['verifier.py', '--net', 'conv6_base', '--spec', 'preliminary_test_cases/conv6_base/img0_cifar10_0.000593.txt']
main()