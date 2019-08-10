"""
Figure: Yi--A--B--Xi

For each A--B coupling in train df, we find what angles all other atoms Yi make from A--B bond on A as center.
Similarly we find Xi with B as center. We discritize the angle so as to get conical regions. We aggregate within a
conical region
Aggregation is done of
    1. Induced charge.(Electronegativity diff)
    2. Mass
    3. Count
    4. Lone pairs.
    5. Electron count in outermost orbit.

"""
import numpy as np


def compute_feature_one_molecule(train_data_ai_0: np.array, train_data_ai_1: np.array, molecule_data: np.array):
    """
    molecule_data: Columns are x,y,z, Induced charge, mass, #lone pairs, #electrons in outermost shell. It is ordered by
                atom_index
    train_data_ai_0: Columns are x,y,z, Induced charge, mass, #lone pairs, #electrons in outermost shell.
                    For atom_index_0
    train_data_ai_1: Columns are x,y,z, Induced charge, mass, #lone pairs, #electrons in outermost shell.
                    For atom_index_1

    """
    atom_count = molecule_data.shape[0]
    train_count = train_data_ai_0.shape[0]

    xyz = [0, 1, 2]
    bond_vector_0 = train_data_ai_0[:, xyz] - train_data_ai_1[:, xyz]
    bond_vector_0 = bond_vector_0 / np.linalg.norm(bond_vector_0, axis=1).reshape(train_count, 1)
    # train_data_ai_0 = np.concatenate([train_data_ai_0, bond_vector], axis=1)

    # (#train_df.shape[0],features,replicas)
    A = np.tile(bond_vector_0.T, (atom_count, 1, 1)).T

    C = np.tile(train_data_ai_0[:, xyz].T, (atom_count, 1, 1)).T
    B = np.tile(molecule_data.T, (train_count, 1, 1))

    B[:, xyz, :] = B[:, xyz, :] - C[:, xyz, :]
    YiA_distance = np.linalg.norm(B[:, xyz, :], axis=1)
    # Ignore self atoms by making the distance very large.
    YiA_distance = np.where(YiA_distance == 0, 1e10, YiA_distance)
    # xyz will become unit vectors, features will get normalized by distance. large distance will have very low
    # contributions
    B = B / YiA_distance.reshape(B.shape[0], 1, B.shape[2])

    angle = (A[:, xyz, :] * B[:, xyz, :]).sum(axis=1)

    # -3 to 3. 7 divisions.
    factor = 3
    angle = (angle * factor).astype(int)

    features = []
    for region in range(-factor, factor + 1, 1):
        filtr = (angle == region).astype(int)
        filtr = filtr.reshape((B.shape[0], 1, B.shape[2]))
        feature_one_region = (B[:, 3:, :] * filtr).sum(axis=2)
        features.append(feature_one_region)

    return np.hstack(features)
