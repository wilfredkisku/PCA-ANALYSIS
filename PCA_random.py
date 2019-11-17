import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

np.random.seed(43)
####################### Data Generation for mapping of 3-d data into 2-d space #####################

##creating a mean vector centered at origin (0,0,0)
##cov matrix is taken as the identity as it would generate a matrix of varianvce 1 for all variables
mu_vec1 = np.array([0,0,0])
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 30).T
assert class1_sample.shape == (3,30), "The matrix does not have 3x30 dim"

##creating a mean vector centered at (1,1,1)
##cov matrix is taken as the identity as it would generate a matrix of varianvce 1 for all variables
mu_vec2 = np.array([1,1,1])
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 30).T
assert class2_sample.shape == (3,30), "The matrix does not have 3x30 dim"

####################### Plot the 3-d data for visualtization #######################################

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10
ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], '^', markersize=8, alpha=0.5, color='red', label='class2')

plt.title('Samples for class 1 and class 2')
ax.legend(loc='upper right')

plt.show()

####################### A combine all the samples of different classes #############################
##mean of the x,y and  axis values
all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
assert all_samples.shape == (3,60), "The matrix has not the dimensions 3x60"

mean_x = np.mean(all_samples[0,:])
mean_y = np.mean(all_samples[1,:])
mean_z = np.mean(all_samples[2,:])

mean_vector = np.array([[mean_x],[mean_y],[mean_z]])

print('Mean Vector:\n', mean_vector)

cov_mat = np.cov([all_samples[0,:],all_samples[1,:],all_samples[2,:]])
print('Covariance Matrix:\n', cov_mat)

eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

print('Eigven values of covariance matrix:\n',eig_val_cov)
print('Eigven vectors of covariance matrix:\n',eig_vec_cov)

#matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
#print('Matrix W:\n', matrix_w)

for ev in eig_vec_cov:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

print('Sorting the eigenvalues in the ascending order:')
# Visually confirm that the list is correctly sorted by decreasing eigenvalues
for i in eig_pairs:
    print(i[0])

matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
print('Matrix W:\n', matrix_w)

transformed = matrix_w.T.dot(all_samples)
assert transformed.shape == (2,60), "The matrix is not 2x60 dimensional."

plt.plot(transformed[0,0:30], transformed[1,0:30], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(transformed[0,30:60], transformed[1,30:60], '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.xlabel('transformed x_values')
plt.ylabel('Transformed y_values')
plt.legend()
plt.title('Transformed samples with class labels')

plt.show()
