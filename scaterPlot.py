import matplotlib.pyplot as plt

fp = open("Assignment1_Q4_data.txt", 'r')
X1, Y1 = [0 for i in range(500)], [0 for i in range(500)]
X2, Y2 = [0 for i in range(500)], [0 for i in range(500)]

for i in range(500):
    X1[i], Y1[i] = map(float, fp.readline().split())

for i in range(500):
    X2[i], Y2[i] = map(float, fp.readline().split())


plt.scatter(X1, Y1, label='Class A', color='blue', marker="*")
plt.scatter(X2, Y2, label='Class B', color='red', marker=".")

plt.xlabel('x')
plt.ylabel('y')

plt.legend()
plt.show()