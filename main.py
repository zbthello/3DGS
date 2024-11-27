from PIL import Image
from utils.general_utils import PILtoTorch
def fibonacci(n):
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result

if __name__ == '__main__':
    print("hello---3DGS")
    print(fibonacci(6))




