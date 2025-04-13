import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd
np.random.seed(42)

# Loading the dataset
dataset = load_dataset("ylecun/mnist")
data = dataset["train"]
images_ = np.array(data["image"])
digits_ = np.array(data["label"])


# Sampling 1000 random images, 100 each of each digit (0-9)
indices = {}
for d in range(10):
    indices[d] = np.where(digits_ == d)[0]
images, digits = [], []
for d in range(10):
    random_indices = np.random.choice(indices[d], 100, replace=False)
    for i in random_indices:
        images.append(images_[i])
        digits.append(d)
images = np.array([np.reshape(i, -1) for i in images])


# Q1.(i) Implementing the PCA algorithm

# function to calculate the principal components
def pca(images):
    m, n = images.shape
    mean = np.zeros(n)
    for i in range(n):
        mean[i] = np.sum(images[:, i]) / m
    images = images - mean                 # centering the dataset
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov_matrix[i, j] = np.sum(images[:, i] * images[:, j]) / (m)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)       # calculating eigenvalues and eigenvectors of the covariance matrix
    indices = np.argsort(eigen_values)[::-1]             
    eigen_values = eigen_values[indices]                        # sorting the eigen values in descending order
    eigen_vectors = eigen_vectors[:, indices]
    return eigen_values, eigen_vectors, mean



eigen_values, eigen_vectors,mean = pca(images)
print(f"Total number of dimensions in each image: {images[0].shape[0]}")
print("The number of significant principal components:", (eigen_values>0).sum())



# Plotting the top 30 principal components of the dataset
fig, axes = plt.subplots(5, 6, figsize=(9, 7)) 
axes = axes.ravel()
for i in range(30):
    pc = eigen_vectors[:, i].reshape(28, 28) 
    axes[i].imshow(pc, cmap="gray")
    axes[i].set_title(f"{i + 1}", fontsize=8)
    axes[i].axis("off")
plt.suptitle("Top 30 Principal Components")
plt.show()


total_var = np.sum(eigen_values)  # calculating total variance of the dataset
var_ratio = []
for e in eigen_values:
    var_ratio.append((e / total_var))
var_ratio = np.array(var_ratio)
var_ratio*=100
cumulative_var = np.cumsum(var_ratio)       # calculating cumulative variance of the dataset


components = [3,5, 10, 25, 50, 75, 100, 125,134, 150, 175, 200,300, 400,500, 662,784]
var = []
for c in components:
    var.append(cumulative_var[c - 1])


# table which shows the amount of variance explained by different number of principal components
print("This table shows the amount of variance explained by different number of principal components:")
df1= pd.DataFrame({
    "PC's": components,
    "Variance(%)": var
})
print(df1.to_string(index=False))

print(f"Number of components explaining 95% variance: {np.argmax(cumulative_var >= 95) + 1}")


# table which shows the amount of variance explained by the top 20 eigenvectors

print("This table shows the amount of variance explained by each of the top 20 principal components:")
df2= pd.DataFrame({
    "Principal Component No": np.arange(1, 21),
    "Variance Explained (%)": var_ratio[:20]
})
print(df2.to_string(index=False))



# Q1.(ii) Reconstruction of the dataset

# function for reconstruction of the dataset
def reconstruct_images(images, dim):
    _, eigenvectors, mean = pca(images)
    images = images -mean
    imgs_recon = []
    for d in dim:
        d_eigenvectors = eigenvectors[:, :d]
        img_recon = ((images @ d_eigenvectors) @ d_eigenvectors.T)+mean
        imgs_recon.append(img_recon)
    return imgs_recon


# Plot which shows the reconstruction of the dataset using different dimensionality representations
dim = [5,10,25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700]
plt.figure(figsize=(5.5, 5.5))
plt.subplot(4, 4,1)
plt.imshow(images[301].reshape(28, 28), cmap='gray')
plt.title("Original", fontsize=10)
plt.axis('off')
imgs_recon = reconstruct_images(images, dim)
for i in range(len(dim)):
    plt.subplot(4, 4, i + 2)
    plt.imshow(imgs_recon[i][301].reshape(28, 28), cmap='gray')
    plt.title(f"{dim[i]}", fontsize=7)
    plt.axis('off')

plt.suptitle("Original & Reconstructed Images (Using Principal Components)", fontsize=12)
plt.show()

# Calculating the reconstruction error
errors = []
d = [5,10,25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700]
for i in range(len(d)):
    error = np.mean((images - imgs_recon[i]) ** 2)
    errors.append(error)

# Plot of reconstruction error vs number of principal components
plt.figure(figsize=(6, 4))
plt.plot(d, errors,'-sg')
plt.title("Reconstruction Error vs Number of Principal components", fontsize=14)
plt.xlabel("Principal components")
plt.ylabel("MSE")
plt.show()