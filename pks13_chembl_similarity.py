#!/usr/bin/env python3
import csv
import sys
import time
from urllib.parse import quote

import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, DataStructs

QUERIES = {
    'A1038': 'COc1ccc(-c2noc(C(=O)NCCc3ccc(C(=O)N4CCC(N)C4)cc3)n2)cc1OC',
    'A1375': 'COC1=CC(c2noc(C(=O)NCCc3ccc(C(=O)N4CCC(O)C4)cc3)n2)=C1C',
    'V1123': 'COc1cc(-c2nc(C(=O)NCCc3ccc(C(=O)N4CC(OC)C4)cc3)no2)ccc1CO',
}
THRESHOLDS = (70, 50)  # Lower to 50 only when no 70%-threshold hits are returned
MAX_RESULTS_PER_QUERY = 100
TOP_HITS_TO_EXPORT = 10
BASE = 'https://www.ebi.ac.uk/chembl/api/data/similarity/'
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
HEADERS = ['query_id', 'query_smiles', 'hit_rank', 'chembl_id', 'tanimoto_rdkit',
           'hit_smiles', 'search_threshold', 'chembl_url', 'catalog_status', 'notes']

def get_hits(session, smiles, threshold):
    url = f'{BASE}{quote(smiles, safe="")}/{threshold}.json'
    collected = []
    params = {'limit': min(MAX_RESULTS_PER_QUERY, 100), 'offset': 0}
    while len(collected) < MAX_RESULTS_PER_QUERY:
        response = session.get(url, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        batch = payload.get('molecules', [])
        collected.extend(batch)
        if not batch or len(batch) < params['limit']:
            break
        params['offset'] += len(batch)
        time.sleep(0.2)
    return collected[:MAX_RESULTS_PER_QUERY]

def main():
    session = requests.Session()
    session.headers.update({'User-Agent': 'Pks13-study-similarity/1.0 (academic research)'})
    results = []
    errors = []
    for name, query in QUERIES.items():
        molecule = Chem.MolFromSmiles(query)
        if molecule is None:
            errors.append(f'{name}: invalid query SMILES')
            continue
        canonical = Chem.MolToSmiles(molecule)
        query_fp = FP.GetFingerprint(molecule)
        hits = []
        used_threshold = None
        for cutoff in THRESHOLDS:
            try:
                hits = get_hits(session, canonical, cutoff)
                used_threshold = cutoff
            except requests.RequestException as exc:
                errors.append(f'{name}: request failed at threshold {cutoff}: {exc}')
                break
            if hits:
                break
        scored = []
        seen = set()
        for hit in hits:
            hit_id = hit.get('molecule_chembl_id', '')
            hit_smiles = (hit.get('molecule_structures') or {}).get('canonical_smiles')
            if not hit_id or not hit_smiles or hit_id in seen:
                continue
            seen.add(hit_id)
            hit_mol = Chem.MolFromSmiles(hit_smiles)
            if hit_mol is None:
                continue
            sim = DataStructs.TanimotoSimilarity(query_fp, FP.GetFingerprint(hit_mol))
            scored.append((sim, hit_id, hit_smiles))
        scored.sort(reverse=True)
        for rank, (sim, hit_id, hit_smiles) in enumerate(scored[:TOP_HITS_TO_EXPORT], 1):
            results.append({
                'query_id': name, 'query_smiles': canonical, 'hit_rank': rank,
                'chembl_id': hit_id, 'tanimoto_rdkit': f'{sim:.4f}',
                'hit_smiles': hit_smiles, 'search_threshold': used_threshold,
                'chembl_url': f'https://www.ebi.ac.uk/chembl/compound_report_card/{hit_id}/',
                'catalog_status': 'NOT VERIFIED',
                'notes': 'ChEMBL record is not evidence of commercial availability or Pks13 activity',
            })
        print(f'{name}: {len(scored)} valid ChEMBL hits retrieved; top {min(len(scored), TOP_HITS_TO_EXPORT)} exported')
    with open('pks13_chembl_hits.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(results)
    if errors:
        with open('pks13_search_errors.txt', 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(errors) + '\n')
        print('Some searches failed; see pks13_search_errors.txt', file=sys.stderr)
    print('Saved pks13_chembl_hits.csv (empty if no hits found or search failed)')

if __name__ == '__main__':
    main()
