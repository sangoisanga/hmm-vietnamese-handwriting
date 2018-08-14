from itertools import product

if __name__ == '__main__':
    a = [1, 2, 3]
    b = [7, 8, 9]
    c = [4, 5, 6]
    #print list(product(a, b, repeat=1))
    temp = list(eval("product(a,b,repeat=1)"))
    print temp
