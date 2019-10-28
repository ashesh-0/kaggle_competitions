# Info on data
1. train_csv: (molecule, two atoms, type and scalar_coupling_constant).
2. structures: (molecule, atom, atom type, xyz coordinates)

In competition, it is said that using Quantum mechanics, one infers the target value using just above information (atom type and atom co-ordinate information). So the problem is then to find the
correct representation for a bond which encapsulates different properties of its neighbouring atoms.

# Summary of what was done
In the final model, I used only train.csv and structures.csv. All other data could not be used to get a better model and hence was discarded.

## Model Structure
3 models were created which were then stacked to get the final model.
1. **Catboost** (LB: -1.95): It is GBDT (tree based). In the training data for it, each row contained one pair of atoms for which the target was known. Note that two bonds of a molecule were independently represented in the training data.
2. **LightGBM** (LB: -1.87): It is also GBDT (tree based). It varies from Catboost in the way the tree is built. In Catboost at all intermediate nodes at one level of a tree use the same feature to split. In LightGBM, tree is built one node at a time. So LightGBM is 'stronger' but at the same time more prone to overfitting and hyperparameter sensitivity. Training data is identical to Catboost model.
3. **GNN-Graph neural networks** (LB: -2.05): Here, one row of training data comprises of features for all bonds present in the molecule. Infact, it had the features for every pair of atoms present in the molecule. Ofcourse, for those atom pairs which were far away from each other and did not affect each other chemically, their features did not effect the optimization process. Note that here we have the information about which set of bonds belong to one molecule. This was not present in tree based models where each bond was treated as independent training example. Also, neighboring bond information is implicit in the way this model learns. So there is no need to create features like 'what is the mass of 2nd neighbor to this atom' as long as you have the feature 'mass of the atom'. Information about neighbors's features is passed to the atom in the learning process.

## Features for Catboost and LightGBM
**Bond detection**

Initially I inferred bonds([Code](find_edges.py)). However, I later used openbabel for bond detection. See relevant section below for details on openbabel.

**Neighbor detection**

I defined neighbors in 5 ways:
1. Using the bond detection module, we got the edge data. Using it, I was able to define 1st neighbor atoms, 2nd neighbor atoms and so on. I was also able to figure out intermediate atoms in path from atomindex0 to atomindex1. ([Code](compute_intermediate_atoms.py#L121))
2. Using distance between atoms and standard bond length values of different bonds, neighbors were defined to be those pair of atoms which were close enough to be considered to be bonded ([Code](neighbor_features_distance.py#L104))
3. Based on atomindex. For example, neighbors of atomindex 5 will be 3,4,6,7. ([Code](neighbor_features_atom_index.py))
4. k-nearest atoms were considered as neighbors. ([Code](edge_features.py#L182))
5. I divided the region around center of the bond by creating right circular cones with different base angles. Cone's axis was the bond. It gave me neighbors which were in different angular neighborhood ([0°-60°], (60°-120°) and so on). ([Code](conical_segmented_features.py))

**Neighbor based features**

I created aggregate of Electronegativity, Valency, Lone pairs, bond angles,mass and distance for neighbors generated from above 5 ways.

**Angle based features**

a. Dihedral angle and some other angles were also computed for 2,3 bond apart atom pairs. (3JHH,3JHN,3JHC,2JHH,2JHN,2JHC)
b. Aggregate of angles for neighbors.

**Estimation of partial charge**

Using electronegativity information and co-ordinates of atoms, I computed electronegativity vectors normalized by distance between the atoms. Doing a vector sum of these electronegativity vectors over bond neighbors yielded me an estimate of partial charge on each atom. ([Code](edge_features.py#L95)). Later in the competition, I started to use sum of electronegativity difference normalized by distance as a measure of partial charge (no vector addition).

**Potential based features**

Used yukawa,coulomb potential using the estimated charge. ([Code](atom_potentials.py#L14)).

**Other features**

I used some of the atom features generate by openbabel. Using edge data, I was able to also infer cycle length and other related features ([Code](cycle_features.py#L23)). I also used bond energies as feature between bonds of the molecule([Code](bond_features.py#L31)). Carbon hybridization was used as a feature. For 3 bond distant atom pairs which had 2 sp2 hybridized carbon atoms as intermediates, Cis/Trans configuration was used as a feature ([Code](intermediate_atom_features.py#L144)). I also tried to estimate how much pi bonds have electron 'donor' tendency. ([Code](pi_donor.py))

**Features for GNN**

I had to recompute the edge features as they were to be computed between each possible atom pairs in a molecule. I ended up coding up a numpy heavy implementation of the top features again which I had computed for tree based models. Earlier features were computed mostly using pandas.


## Experiments:
1. In linear regression, using 4 models for each component was equivalent to one model (it makes sense in hindsight :().
2. Multiple models, one for each type is better  than one linear regression LB 1.3 -> LB 1.19
3. Catboost is even better LB 1.19 => LB 1.12
4. Using distance features, catboost goes to LB 0.65.
5. Separate model for each type is better. It is so as then we optimize the metric which is specified.
6. Fixing distance feature, I get to now LB 0.26

... There were multiple experiments. Later ones did not get documented. Finally, I was able to get LB -2.05 using single model. :)


## Information from Reading
1. scalar coupling is symmetric with respect to ordering of the two atoms involved. This means that I can replicate the train by swapping atom_index_0 with atom_index_1. However, this replication needs to be done after we split train from validation.
2. possible spin types of all atoms will be helpful.
3. In 3 bond neighbors, trans alkenes have larger coupling constants than sys.
4. Currently, I'm just looking at 1 neighbors. I should be looking at 3 neighbor Hydrogen.


### Openbabel usage:

Link: http://openbabel.org/dev-api/classOpenBabel_1_1OBBond.shtml

```
import openbabel
import pybel
mol = list(pybel.readfile('mol2','dsgdb9nsd_000001.mol2'))[0]
for bond in openbabel.OBMolBondIter( mol.OBMol):
    print(bond.GetIdx(),bond.GetBO(),bond.GetBeginAtom().GetIdx(),bond.GetEndAtom().GetIdx())
```

Bash command to add multiple xyz files into one molecule file. `obabel -ixyz -omol2 structures/dsgdb9nsd_00000*.xyz  -O abcd.mol2 --separate`

