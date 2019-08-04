"""
atom_index of it exactly matches that of structures_df
"""

import openbabel
import os
import pandas as pd
import pybel
from tqdm import tqdm
import numpy as np


def get_atom_feature_columms():
    columns = [
        'atom_index',
        'mol_id',
        'SpinMultiplicity',
        'Valence',
        'HvyValence',
        'Type',
        'PartialCharge',
        'X',
        'Y',
        'Z',
        'MemberOfRingCount',
        'MemberOfRingSize',
        'CountRingBonds',
        'SmallestBondAngle',
        'AverageBondAngle',
        'IsHbondAcceptor',
        'HasDoubleBond',
    ]
    dtypes = {
        'atom_index': np.uint8,
        'mol_id': np.uint32,
        'SpinMultiplicity': np.float32,
        'Valence': np.uint8,
        'HvyValence': np.uint8,
        'Type': str,
        'PartialCharge': np.float32,
        'X': np.float32,
        'Y': np.float32,
        'Z': np.float32,
        'MemberOfRingCount': np.uint8,
        'MemberOfRingSize': np.uint8,
        'CountRingBonds': np.uint8,
        'SmallestBondAngle': np.float32,
        'AverageBondAngle': np.float32,
        'IsHbondAcceptor': bool,
        'HasDoubleBond': bool,
    }
    return (columns, dtypes)


def _get_atom_data(mol):
    output = []
    mol_name = os.path.basename(mol.title).split('.')[0]
    for atom in openbabel.OBMolAtomIter(mol.OBMol):
        Id = atom.GetId()
        SpinMultiplicity = atom.GetSpinMultiplicity()
        Valence = atom.GetValence()
        HvyValence = atom.GetHvyValence()
        Type = atom.GetType()
        PartialCharge = atom.GetPartialCharge()
        X = atom.GetX()
        Y = atom.GetY()
        Z = atom.GetZ()
        MemberOfRingCount = atom.MemberOfRingCount()
        MemberOfRingSize = atom.MemberOfRingSize()
        CountRingBonds = atom.CountRingBonds()
        SmallestBondAngle = atom.SmallestBondAngle()
        AverageBondAngle = atom.AverageBondAngle()
        IsHbondAcceptor = atom.IsHbondAcceptor()
        HasDoubleBond = atom.HasDoubleBond()
        data = [
            Id,
            mol_name,
            SpinMultiplicity,
            Valence,
            HvyValence,
            Type,
            PartialCharge,
            X,
            Y,
            Z,
            MemberOfRingCount,
            MemberOfRingSize,
            CountRingBonds,
            SmallestBondAngle,
            AverageBondAngle,
            IsHbondAcceptor,
            HasDoubleBond,
        ]
        output.append(data)
    return output


def get_atom_data(fname):
    output = []
    extension = os.path.basename(fname).split('.')[-1]
    for mol in pybel.readfile(extension, fname):
        output += _get_atom_data(mol)
    return output


def to_hdf(output, outputfname, key, mol_name_encoder):
    columns, dtypes = get_atom_feature_columms()
    df = pd.DataFrame(output, columns=columns)
    df['mol_id'] = mol_name_encoder.transform(df['mol_id'])
    df = df.astype(dtypes)
    df.to_hdf(outputfname, key=key, format='table', append=True)


def write_atom_features_to_file(outputfname, hdf_key, data_dir, mol_name_encoder):
    data_files = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir)]

    output = []
    for i, datafname in tqdm(list(enumerate(data_files))):
        output += get_atom_data(os.path.join(data_dir, datafname))
        if i % 10000 == 0 and i > 0:
            to_hdf(output, outputfname, hdf_key, mol_name_encoder)
            output = []

    to_hdf(output, outputfname, hdf_key, mol_name_encoder)
