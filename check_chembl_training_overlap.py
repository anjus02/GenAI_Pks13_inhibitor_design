#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

QUERIES = {
    'CHEMBL5424265': 'COC1=C(C=C(C=C1)C2=NOC(=N2)C(=O)NCCC3=CC=C(C=C3)C(=O)N4CC(C4)F)OC',
    'CHEMBL5394169': 'COC1=C(C=C(C=C1)C2=NOC(=N2)C(=O)NCCC3=CC=C(C=C3)C(=O)N4CCC4)OC',
    'CHEMBL5416518': 'COC1CN(C1)C(=O)C2=CC=C(C=C2)CCNC(=O)C3=NC(=NO3)C4=CC(=C(C=C4)OC)OC',
}

def get_rows(path, sheet):
    ext = path.suffix.lower()
    if ext in ('.csv', '.tsv'):
        with path.open(encoding='utf-8-sig', newline='') as f:
            yield from csv.DictReader(f, delimiter='\t' if ext == '.tsv' else ',')
    elif ext in ('.xlsx', '.xlsm'):
        from openpyxl import load_workbook
        book = load_workbook(path, read_only=True, data_only=True)
        ws = book[sheet] if sheet else book.active
        itr = ws.iter_rows(values_only=True)
        headers = [str(v).strip() if v is not None else '' for v in next(itr)]
        for values in itr:
            yield dict(zip(headers, values))
        book.close()
    elif ext in ('.smi', '.smiles', '.txt'):
        with path.open(encoding='utf-8-sig') as f:
            for i, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts or parts[0].startswith('#') or parts[0].lower() == 'smiles':
                    continue
                yield {'SMILES': parts[0], 'ID': parts[1] if len(parts) > 1 else str(i)}
    elif ext == '.sdf':
        for i, mol in enumerate(Chem.SDMolSupplier(str(path), removeHs=False), 1):
            if mol is None:
                yield {'SMILES': '', 'ID': str(i)}
            else:
                yield {'SMILES': Chem.MolToSmiles(Chem.RemoveHs(mol)),
                       'ID': mol.GetProp('_Name') if mol.HasProp('_Name') else str(i)}
    else:
        raise ValueError('Supported input: .csv, .tsv, .xlsx, .xlsm, .smi, .smiles, .txt, .sdf')

def get_column(keys, requested, possible):
    if requested:
        match = next((k for k in keys if str(k).lower() == requested.lower()), None)
        if match is None:
            raise ValueError(f'Column {requested!r} not found. Available: {keys}')
        return match
    return next((k for k in keys if str(k).strip().lower().replace(' ', '_') in possible), None)

def prep(smiles):
    mol = Chem.MolFromSmiles(str(smiles).strip()) if smiles else None
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    connectivity = Chem.MolToInchiKey(mol).split('-')[0]
    fp = AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
    return {'smiles': smi, 'connectivity': connectivity, 'fp': fp}

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, type=Path, help='Original 246 Pks13 training compounds / Supplementary S1')
    p.add_argument('--smiles-column', help='SMILES column, auto-detected otherwise')
    p.add_argument('--id-column', help='Compound ID column, auto-detected otherwise')
    p.add_argument('--sheet', help='Excel sheet name; defaults to first sheet')
    p.add_argument('--output', default='chembl_training_overlap.csv')
    args = p.parse_args()
    rows = list(get_rows(args.dataset, args.sheet))
    if not rows:
        raise SystemExit('Dataset has no rows.')
    keys = list(rows[0].keys())
    smi_key = get_column(keys, args.smiles_column, {'smiles','canonical_smiles','canonicalsmiles','isomeric_smiles','smiles_string','mol_smiles'})
    id_key = get_column(keys, args.id_column, {'id','compound_id','molecule_id','chembl_id','name','compound','molecule_chembl_id'})
    if not smi_key:
        raise SystemExit(f'Cannot detect a SMILES column. Available columns: {keys}\nPass --smiles-column COLUMN_NAME')
    records = []
    invalid = []
    for i, row in enumerate(rows, 1):
        item = prep(row.get(smi_key))
        if item is None:
            invalid.append(i)
            continue
        item['id'] = str(row.get(id_key) or f'row_{i}') if id_key else f'row_{i}'
        item['row'] = i
        records.append(item)
    if not records:
        raise SystemExit('No valid molecular SMILES in supplied dataset.')

    output = []
    for chembl_id, query_smiles in QUERIES.items():
        q = prep(query_smiles)
        if q is None:
            raise SystemExit(f'Invalid query SMILES: {chembl_id}')
        exact = [r for r in records if r['smiles'] == q['smiles']]
        connectivity = [r for r in records if r['connectivity'] == q['connectivity']]
        ranks = sorted(((DataStructs.TanimotoSimilarity(q['fp'], r['fp']), r) for r in records),
                       key=lambda item: item[0], reverse=True)
        top_sim, top = ranks[0]
        outcome = {
            'chembl_id': chembl_id,
            'exact_canonical_smiles_match': 'YES' if exact else 'NO',
            'connectivity_inchikey_match': 'YES' if connectivity else 'NO',
            'matching_dataset_ids': '; '.join(r['id'] for r in connectivity),
            'matching_dataset_rows': '; '.join(str(r['row']) for r in connectivity),
            'nearest_dataset_id': top['id'],
            'nearest_dataset_row': top['row'],
            'nearest_morgan_tanimoto': f'{top_sim:.4f}',
            'nearest_dataset_smiles': top['smiles'],
            'query_canonical_smiles': q['smiles'],
        }
        output.append(outcome)
        print(f'\n{chembl_id}: exact SMILES={outcome["exact_canonical_smiles_match"]}; '
              f'connectivity={outcome["connectivity_inchikey_match"]}')
        print(f'  Matched IDs: {outcome["matching_dataset_ids"] or "none"}')
        print(f'  Nearest training compound: {top["id"]}; Tanimoto={top_sim:.4f}')
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f'\nRead {len(rows)} rows; parsed {len(records)} molecules; invalid/blank rows: {len(invalid)}')
    if invalid:
        print('Invalid/blank row indices (data rows starting at 1):', invalid[:30])
    print('Saved:', args.output)
    print('Note: an InChIKey first-block match indicates identical connectivity, not necessarily identical stereochemistry.')

if __name__ == '__main__':
    main()
