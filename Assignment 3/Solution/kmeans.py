import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv(r'cm_dataset_2.csv').values           # loading the dataset

# Q2.(i) Implementation of Llyod's algorithm

# function to calculate error between the means and the assignments at each iteration
def calculate_error(X, means, indicators, k):
    error = 0
    for j in range(k):
        points = X[indicators == j]
        error += np.sum(np.linalg.norm(points - means[j], axis=1))
    return error

# function to implement the Llyod's algorithm
def Llyods(X, k,rs):
    np.random.seed(rs)
    n, d = X.shape
    means = X[np.random.choice(n, k, replace=False)]
    indicators = np.zeros(n)
    errors = []
    iteration = 0
    while True:
        iteration += 1
        for i in range(n):
            pt = X[i]
            distances = np.linalg.norm(pt - means, axis=1)
            indicators[i] = np.argmin(distances)
        new_means = np.empty((k, d))
        for j in range(k):
            pts = X[indicators== j]
            if len(pts) > 0:
                new_means[j] = np.mean(pts, axis=0)
        error = calculate_error(X, means, indicators, k)
        errors.append(error)
        if np.linalg.norm(new_means - means) < 1e-6:
            break
        means = new_means
    return means, indicators, errors, iteration       


# function to plot the cluster regions for k=2 and for different initializations of the means
def plot_graph(X,i,errors,indicators, means):
    figure, ax = plt.subplots(1, 2, figsize=(10, 5))
    figure.suptitle(f'Initialization {i + 1}', fontsize=16, fontweight='bold')
    ax[0].plot(errors,'-g')
    ax[0].set_xlabel('Number of Iterations',fontsize=11)
    ax[0].set_ylabel('Error',fontsize=11)
    ax[0].set_title('Plot of Error vs Number of Iterations',fontsize=13)
    colors = ["plum","orange"]
    for j in range(2):
        clstr_pts = X[indicators == j]
        ax[1].scatter(clstr_pts[:, 0], clstr_pts[:, 1],color = colors[j], label=f'Cluster {j + 1}')
    ax[1].scatter(means[:, 0], means[:, 1], color='red', marker='x',s=150, label='Center')
    ax[1].set_title('Cluster Regions')
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()




k = 2
results = []
for init_num in range(5):
    rs = [1,8,5,3,7]          # usind different values of random state for different initializations of the means
    means, indicators, errors, iterations = Llyods(dataset,2,rs[init_num])
    error = errors[-1]
    plot_graph(dataset, init_num,errors,indicators,means)
    results.append({"Initialization": init_num + 1, "Final Error": error, "Iterations": iterations})


# dataframe which stores the values of final errors and number of iterations for each initialization
print("Table to show the final errors and number of iterations taken for different initializations of the means:")
df = pd.DataFrame(results)
print(df.to_string(index=False))


# plot of number of iterations based on different initializations
plt.figure(figsize=(6, 4))
plt.bar(df["Initialization"], df["Iterations"], color='skyblue', edgecolor='black',width = 0.5)
plt.xticks(df["Initialization"])
plt.yticks(np.arange(0, df["Iterations"].max() + 3, 4)) 
plt.xlabel("Initialization",fontsize=12)
plt.ylabel("Number of Iterations",fontsize=12)
plt.title("Number of Iterations based on Initialization",fontsize=15)
plt.show()


# Q2.(ii) Plotting the Voronoi regions
for k in [2, 3, 4, 5]:
    means, _, _, _ = Llyods(dataset,k,42)
    xx, yy = np.meshgrid(np.linspace(-16, 16, 1000), np.linspace(-16, 6, 1000))
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    distances = []
    for pt in grid:
        dist = np.linalg.norm(pt - means, axis=1)
        distances.append(dist)
    pt_labels = np.argmin(distances, axis=1)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, pt_labels.reshape(xx.shape), cmap="Accent", alpha=0.6)
    plt.contour(xx, yy, pt_labels.reshape(xx.shape), colors='black', linewidths=1)
    plt.scatter(dataset[:, 0], dataset[:, 1], color="blue", s=15, label="Data-points")  
    plt.scatter(means[:, 0], means[:, 1], color="black", marker="X", s=150, label="Cluster Means/Centers")
    plt.title(f"Voronoi Regions (K={k})",fontsize=14,fontweight='bold')
    plt.legend()
    plt.show()


