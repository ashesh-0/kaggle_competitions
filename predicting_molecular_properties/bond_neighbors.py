def bond_neighbor_features(X_df, str_df):
    """
    1. How many neighbors are present for both indices.
    2. Average bond length
    3. How average bond length compares with the bond in consideration.
    """
    assert len(
        set(['molecule_id', 'atom_1', 'atom_0', 'x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']) - set(X_df.columns)) == 0
    assert 'molecule_id' in str_df.columns
