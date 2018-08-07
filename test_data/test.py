from itertools import product

if __name__ == '__main__':
    a = [1, 2, 3]
    b = list(product(a, repeat=3))
    print b
    print len(b)
