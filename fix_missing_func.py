import json

# 1. Extract smiles_to_graph_hybrid from Corr_SRP_ChemSpace.ipynb
with open('Corr_SRP_ChemSpace.ipynb', 'r', encoding='utf-8') as f:
    nb_orig = json.load(f)

missing_func_src = ""
for cell in nb_orig['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        start_idx = src.find('def smiles_to_graph_hybrid(')
        if start_idx >= 0:
            missing_func_src = src[start_idx:]
            break

if not missing_func_src:
    print("Could not find smiles_to_graph_hybrid in Corr_SRP_ChemSpace.ipynb")
else:
    # 2. Append it to Cell 10 in Corr_SRP_Optuna.ipynb and Irrit_SRP_Optuna.ipynb
    for nb_file in ['Corr_SRP_Optuna.ipynb', 'Irrit_SRP_Optuna.ipynb']:
        try:
            with open(nb_file, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            
            for i, cell in enumerate(nb['cells']):
                if cell['cell_type'] == 'code':
                    src = ''.join(cell['source'])
                    if 'def train_with_early_stopping' in src:
                        # Append the missing function if not already there
                        if 'def smiles_to_graph_hybrid' not in src:
                            cell['source'] = [src + '\n\n' + missing_func_src]
                            print(f"Appended smiles_to_graph_hybrid to {nb_file}")
                        break
            
            with open(nb_file, 'w', encoding='utf-8') as f:
                json.dump(nb, f, ensure_ascii=False, indent=1)
                
        except Exception as e:
            print(f"Error processing {nb_file}: {e}")
