import numpy as np
import cv2
import sys 


def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

if __name__ == '__main__':
    
    file_img = sys.argv[1]

    #obtain the orginal image from the command line
    X_raw = cv2.imread(file_img,0)
    
    #display the original image
    cv2.namedWindow('Display',0)
    cv2.imshow('Display',X_raw)
    cv2.waitKey(0)
    
    #centering the row pixel values 
    X_cen = (X_raw.T - np.mean(X_raw,axis=1)).T
    
    #extracting the eigen values and the eigen vectors 
    [eig_val, eig_vec] = np.linalg.eig(np.cov(X_cen))

    print 'eig_values=\n',eig_val
    print 'eig_val_shape=\n',eig_val.shape

    print 'eig_vector=\n',eig_vec
    print 'eig_vec_shape=\n',eig_vec.shape

    #sort the eigenvalues in the descending order
    idx_sorted = np.argsort(eig_val)[::-1]

    #sorting the eigenvectors and taking k components from the eigenvectors 
    k = 5
    
    eig_vec = eig_vec[:, idx_sorted]
    eig_vec = eig_vec[:, range(k)]

    scores = np.dot(X_cen,eig_vec)

    #changes made
    #X_cap_red = (scores.T + np.mean(X_raw,axis=1)).T
    #X_cap_red = scale(X_cap_red,0,255)
    #X_cap_red = np.ndarray.astype(X_cap_red,dtype='uint8')

    #cv2.namedWindow('Display_conv',0)
    #cv2.imshow('Display_conv',X_cap_red)
    #cv2.waitKey(0)

    print 'scores=\n', scores

    X_cap = np.dot(scores, eig_vec.T)

    print 'reconstructed=\n',X_cap
    print 'reconstructed_shape=\n',X_cap.shape

    X_cap_raw = (X_cap.T + np.mean(X_raw,axis=1)).T

    print 'recon_cons_cap=\n',X_cap_raw
    
    X_cap_raw = scale(X_cap_raw,0,255)
    X_cap_raw = np.ndarray.astype(X_cap_raw,dtype='uint8')
    cv2.namedWindow('Display_reconstructed',0)
    cv2.imshow('Display_reconstructed',X_cap_raw)
    cv2.waitKey(0)
