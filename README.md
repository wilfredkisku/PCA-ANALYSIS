# PCA Analysis (Dimensionality Reduction)
The goal of this assignment is to carry out dimension reduction using the concept of Principal Component Analysis (PCA). The sub goals include operations to be carried out dimensionality reduction on particular data such as visual data or images. Also, to carry out feature reduction on a randomly generated dataset to
show a visual mapping of data from 3D to 2D plane or similarly can be extended from a 2D plane to a 1D or any other higher dimesional planes into lower dimensions space.

## Algorithm
<ul>
<li>Take the whole dataset consisting of d-dimensional samples ignoring the class labels. In the case of the generated samples, I have created a dataset matrix of definite mean and variance that uses the inbuilt function <tt>np.random.multivariate_normal()</tt> function for generating the data.</li>
<li>Compute the d-dimensional mean vector (i.e., the means for every dimension of the whole dataset). Mean centering is essential for performing Principal Component Analysis, as it gives direction of variability across the mean of the samples by creating the covariance matrix.</li>
<li>Compute the scatter matrix (alternatively, the covariance matrix) of the whole data set.</li>
<li>Compute eigenvectors (e1, e2, e3, ...) and corresponding eigenvalues (λ1, λ2, λ3, ...).</li>
<li>Sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors with the largest eigenvalues.</li>
<li>form a d × k dimensional matrix W(where every column represents an eigenvector).</li>
<li>Use this d × k eigenvector matrix to transform the samples onto the new subspace. This can be summarized by the mathematical equation: y = W<sup>T</sup> × x (where x is a d × 1-dimensional vector representing one sample, and y is the transformed k × 1-dimensional sample in the new subspace.)</li>
</ul>

## Result

Data Compression from d dimension to k dimensions


Status: Completed
