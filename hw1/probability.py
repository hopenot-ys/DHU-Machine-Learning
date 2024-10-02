import seaborn as sns
import matplotlib.pyplot as plt

data = [
    2.4, 2.9, 3.0, 3.4, 3.5, 3.6, 3.7, 3.7, 3.8,
    3.8, 3.8, 4.0, 4.1, 4.2, 4.3, 4.8, 4.9, 4.9,
    4.9, 5.0, 5.0, 5.1, 5.2, 5.2, 5.3, 5.4, 5.4,
    5.5, 5.7, 5.7, 5.8, 5.8, 5.8, 5.8, 5.9, 5.9,
    6.0, 6.1, 6.2, 6.2, 6.4, 6.8, 6.8, 6.9, 6.9,
    7.1, 7.2, 7.9, 8.7
]

sns.kdeplot(data, fill=True)

plt.title('Probability Density Estimate (KDE)')
plt.xlabel('Value')
plt.ylabel('Density')

plt.show()
