import json
import glob

notebooks = glob.glob("*.ipynb")

for nb_file in notebooks:
    try:
        with open(nb_file, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        modified = False
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                new_source = []
                for line in cell['source']:
                    # Fix torchvision mismatch issue
                    if '!pip install torch==2.3.0' in line and 'torchvision' not in line:
                        line = line.replace('!pip install torch==2.3.0', '!pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0')
                        modified = True
                    
                    # Standardize umap import
                    if 'import umap.umap_ as umap' in line:
                        line = line.replace('import umap.umap_ as umap', 'import umap')
                        modified = True
                        
                # Update cell source if modified
                if modified:
                    # Actually we need to loop correctly to replace the lines
                    pass
                    
        # Let's do it properly
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                for i in range(len(cell['source'])):
                    if '!pip install torch==2.3.0' in cell['source'][i] and 'torchvision' not in cell['source'][i]:
                        cell['source'][i] = cell['source'][i].replace('!pip install torch==2.3.0', '!pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0')
                        modified = True
                    if 'import umap.umap_ as umap' in cell['source'][i]:
                        cell['source'][i] = cell['source'][i].replace('import umap.umap_ as umap', 'import umap')
                        modified = True
                        
        if modified:
            with open(nb_file, 'w', encoding='utf-8') as f:
                json.dump(nb, f, ensure_ascii=False, indent=1)
            print(f"Fixed {nb_file}")
            
    except Exception as e:
        print(f"Error processing {nb_file}: {e}")
