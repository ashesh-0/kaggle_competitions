# Info on data
1. train_csv: (molecule, two atoms, type and scalar_coupling_constant)
2. structures: (molecule, atom, atom type, xyz coordinates)

Using Quantum mechanics, one gets the target value using just above information.

## Train specific data
1. dipole_moments (charge distribution in molecule): molecule, dipole moment(xyz)
2. magnetic_shielding_tensors (molecule, atom, tensor)
3. mulliken_charges: (molecule, atom, mulliken charge)
4. potential_energy: (molecule, potential energy)
5. scalar_coupling_contributions: scalar coupling constant are sum of 4 terms. This file contains all terms.(molecule, two atoms, type and scalar_coupling_constants)
    a. Fermi Contact contribution
    b. (sd) Spin-dipolar contribution,
    c. (pso) Paramagnetic spin-orbit contribution
    d. (dso) Diamagnetic spin-orbit contribution.


## Model level Ideas:
1. Since target is sum of multiple items, I'll choose linear model/neural networks (over trees as they find it difficult to add features).
2. As a baseline, we can train a linear regression model with one hot encoding for all categorical features.
3. We can train 4 models for the 4 terms which comprises the scalar_coupling_constant. We do need to see correlation among them.
4. We can train neural network which predicts dipole moment, magnetic shielding_tensors, mulliken_charges, potential_energy. Or we can get some approximation formula to compute them using just structure. output of these neural network along with hidden state features can be used as input features for the final neural network predicting the scalar coupling constant.

## Feature level Ideas:
1. Mean encodings for different atom types, molecule types.
2. Charge of the atom, Mass of the atom, radius of the atom, electonegativity of the atom.
3. Neighbor atom stats: This can also be done for x,y,z separately for vector quantities. Different distances. Using a cube is better than sphere as filtering is faster.
4. We can deduce whether the molecule has some regular shape: tetrahedral, etc. From here, we can get more features. one more thing to see if geometry division is already present in the molecules. Also what is the use of atom index. Is it to say for every molecule, atom index is just a numbering?
5. density of atoms, volume enclosed by the polygon, atom distribution.
6. Angle of atom.

## Experiments:
1. In linear regression, using 4 models for each component was equivalent to one model (it makes sense in hindsight :().
2. Multiple models, one for each type is better  than one linear regression 1.3 -> 1.19
3. Catboost is even better 1.19 => 1.12
4. Using distance features, catboost goes to 0.65.
5. Separate model for each type is better. It is so as then we optimize the metric which is specified.
6. Fixing distance feature, I get to now 0.26


## Information from Reading
1. scalar coupling is symmetric. This means that I can replicate the train. However, this replication needs to be done
after we split train from validation.
2. possible spin types of all atoms will be helpful.

1. In 3 bond neighbors, trans alkenes have larger coupling constants than sys.
2. Currently, I'm just looking at 1 neighbors. I should be looking at 3 neighbor Hydrogen.


