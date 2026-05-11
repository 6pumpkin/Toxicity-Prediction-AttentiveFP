import json

# 1. Load the Corr Optuna notebook
with open('Corr_SRP_Optuna.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 2. Iterate through all cells and replace 'Corr' related text with 'Irrit' related text
for cell in nb['cells']:
    if cell['cell_type'] == 'code' or cell['cell_type'] == 'markdown':
        new_source = []
        for line in cell['source']:
            # Replace file name
            line = line.replace('Corr.xlsx', 'Irrit.xlsx')
            
            # Replace dataframe variable name
            line = line.replace('corr_df', 'irrit_df')
            
            # Replace filtering and mapping conditions
            line = line.replace("['Corr', 'Neg']", "['Irrit', 'Neg']")
            line = line.replace("x == 'Corr'", "x == 'Irrit'")
            line = line.replace("1: 'Corrosive (Toxic)'", "1: 'Irritant (Toxic)'")
            
            # Replace result paths
            line = line.replace('/Results/Corr/', '/Results/Irrit/')
            
            new_source.append(line)
        cell['source'] = new_source

# 3. Save to a new notebook file
out_file = 'Irrit_SRP_Optuna.ipynb'
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"[OK] Successfully created {out_file} from Corr_SRP_Optuna.ipynb")
