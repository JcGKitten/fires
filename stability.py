import numpy as np

# stability measurment as proposed in "Measurment the Stability of Feature Selection"

def pearson_stability_ij(arr1,arr2):
    d = len(arr1)
    k_i = np.sum(arr1)
    k_j = np.sum(arr2)

    # catch edge cases as proposed in the paper under 4.1
    if (k_i == 0 or k_i == d) and k_i != k_j :
        return 0
    elif (k_j == 0 or k_j == d) and k_i != k_j :
        return 0
    elif (k_i == 0 or k_i == d) and k_i == k_j :
        return 1
    x_hat_i = k_i / d
    x_hat_j = k_j / d
    arr1 = arr1 - x_hat_i
    arr2 = arr2 - x_hat_j
    dividend = 1/d * np.sum(arr1*arr2)
    divisor = np.sqrt(1/d*np.sum(arr1**2))*np.sqrt(1/d*np.sum(arr2**2))
    return dividend/divisor

def stability_factor(selected_ftrs):
   M = len(selected_ftrs)
   sum_stabilities = 0
   for i in range(M):
       for j in range(i+1, M):
           sum_stabilities += pearson_stability_ij(selected_ftrs[i], selected_ftrs[j])
   return 1/(M*(M-1))*sum_stabilities * 2   