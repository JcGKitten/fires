## Further Questions/ Tasks

### Softmax
- [x] softmax derivative
- [x] read softmax glm implementation again
- [x] check where class check happens
- [] is simultaneous calculation for all given observetions possible, might not work
- [] test for loop over observations
    - [] np.nditer()???
    - [] check what happens if you only give one obs
    - [x] do it before softmax call maybe
- [x] obs with same class in one run
- [x] rewrite np.einsum indexes so they match l, j and c
- [x] bring softmax to fires
- [] test calculating weights
    - [x] bring mu and sigma to one vector befor calculating weights
- [] write explanation page
- [x] class probs to fires
- [] mu_init/ sigma_init?
- [] is shuffle even necarrary?

### Regression
- [x] multiple observations
- [] write explanation page
- [] how is be taken responsibility for the target y???
- [] handle case when only one obs is given

### Common tasks
- [] find datasets
- [] try to use binary FIRES

### Other Implementions
#### OFS
- [] built multiclass perceptron
#### OFSGr
#### FSDS
- [] find implementation
#### EFS
- [] find implementation 
- [] multiclass winnow