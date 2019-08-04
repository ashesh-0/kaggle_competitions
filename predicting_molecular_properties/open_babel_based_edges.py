import openbabel
import pybel
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


def get_edge_columns():
    cols = [
        'mol_id', 'atom_index_0', 'atom_index_1', 'BondLength', 'EqubBondLength', 'BondOrder', 'IsAromatic', 'IsInRing',
        'IsRotor', 'IsAmide', 'IsPrimaryAmide', 'IsSecondaryAmide', 'IsTertiaryAmide', 'IsEster', 'IsCarbonyl',
        'IsSingle', 'IsDouble', 'IsTriple', 'IsClosure', 'IsUp', 'IsDown', 'IsWedge', 'IsHash', 'IsWedgeOrHash',
        'IsCisOrTrans', 'IsDoubleBondGeometry'
    ]
    col_types = {
        'mol_id': np.uint32,
        'atom_index_0': np.uint8,
        'atom_index_1': np.uint8,
        'BondLength': np.float16,
        'EqubBondLength': np.float16,
        'BondOrder': np.uint8,
        'IsAromatic': bool,
        'IsInRing': bool,
        'IsRotor': bool,
        'IsAmide': bool,
        'IsPrimaryAmide': bool,
        'IsSecondaryAmide': bool,
        'IsTertiaryAmide': bool,
        'IsEster': bool,
        'IsCarbonyl': bool,
        'IsSingle': bool,
        'IsDouble': bool,
        'IsTriple': bool,
        'IsClosure': bool,
        'IsUp': bool,
        'IsDown': bool,
        'IsWedge': bool,
        'IsHash': bool,
        'IsWedgeOrHash': bool,
        'IsCisOrTrans': bool,
        'IsDoubleBondGeometry': bool,
    }
    return cols, col_types


def _get_edge_data(mol):
    output = []
    mol_name = os.path.basename(mol.title).split('.')[0]
    for bond in openbabel.OBMolBondIter(mol.OBMol):
        atom_index_0 = bond.GetBeginAtomIdx() - 1
        atom_index_1 = bond.GetEndAtomIdx() - 1
        BondLength = bond.GetLength()
        EqubBondLength = bond.GetEquibLength()
        BondOrder = bond.GetBO()
        IsAromatic = bond.IsAromatic()
        IsInRing = bond.IsInRing()
        IsRotor = bond.IsRotor()
        IsAmide = bond.IsAmide()
        IsPrimaryAmide = bond.IsPrimaryAmide()
        IsSecondaryAmide = bond.IsSecondaryAmide()
        IsTertiaryAmide = bond.IsTertiaryAmide()
        IsEster = bond.IsEster()
        IsCarbonyl = bond.IsCarbonyl()
        IsSingle = bond.IsSingle()
        IsDouble = bond.IsDouble()
        IsTriple = bond.IsTriple()
        IsClosure = bond.IsClosure()
        IsUp = bond.IsUp()
        IsDown = bond.IsDown()
        IsWedge = bond.IsWedge()
        IsHash = bond.IsHash()
        IsWedgeOrHash = bond.IsWedgeOrHash()
        IsCisOrTrans = bond.IsCisOrTrans()
        IsDoubleBondGeometry = bond.IsDoubleBondGeometry()
        data = [
            mol_name, atom_index_0, atom_index_1, BondLength, EqubBondLength, BondOrder, IsAromatic, IsInRing, IsRotor,
            IsAmide, IsPrimaryAmide, IsSecondaryAmide, IsTertiaryAmide, IsEster, IsCarbonyl, IsSingle, IsDouble,
            IsTriple, IsClosure, IsUp, IsDown, IsWedge, IsHash, IsWedgeOrHash, IsCisOrTrans, IsDoubleBondGeometry
        ]
        output.append(data)
    return output


def get_edge_data(fname):
    output = []
    extension = os.path.basename(fname).split('.')[-1]
    for mol in pybel.readfile(extension, fname):
        output += _get_edge_data(mol)
    return output


def to_hdf(output, outputfname, key, mol_name_encoder):
    columns, dtypes = get_edge_columns()
    df = pd.DataFrame(output, columns=columns)
    df['mol_id'] = mol_name_encoder.transform(df['mol_id'])
    df = df.astype(dtypes)
    df.index.name = 'bond_id'
    df.to_hdf(outputfname, key, format='table', append=True)


def write_edges_to_file(outputfname, hdf_key, data_dir, mol_name_encoder):
    data_files = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir)]

    output = []
    for i, datafname in tqdm(list(enumerate(data_files))):
        output += get_edge_data(os.path.join(data_dir, datafname))
        if i % 10000 == 0:
            to_hdf(output, outputfname, hdf_key, mol_name_encoder)
            output = []

    to_hdf(output, outputfname, hdf_key, mol_name_encoder)
