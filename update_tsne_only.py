import json

# 원본 노트북 로드
with open('Corr_SRP_ChemSpace.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# '3.8 데이터 분포 및 화학 공간 분석'이 포함된 셀 찾기
start_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source'])
        if '3.8 데이터 분포 및 화학 공간 분석' in src:
            start_idx = i
            break

if start_idx != -1:
    # 해당 셀 이전까지만 유지
    nb['cells'] = nb['cells'][:start_idx]

# 새로 추가할 셀들 (UMAP 제거, t-SNE 중복 제거 및 3D 플롯 추가)
cells_to_add = []

# Cell 1: Markdown
markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.8 데이터 분포 및 화학 공간 분석 (Chemical Space Analysis)\n",
        "t-SNE를 활용하여 훈련 및 테스트 데이터셋의 분자들의 구조적 분포를 2차원 및 3차원 화학 공간에 시각화합니다.\n",
        "\n",
        "**t-SNE (t-Distributed Stochastic Neighbor Embedding)**:\n",
        "- 고차원 데이터(분자 지문)를 2~3차원으로 축소할 때, **가까이 있는 데이터(비슷한 분자)는 계속 가깝게 유지**하도록 맵핑하는 데 특화된 알고리즘입니다.\n",
        "- 독성을 띠는 분자들이 특정 클러스터로 모이는지 시각적으로 확인하기 아주 좋습니다."
    ]
}
cells_to_add.append(markdown_cell)

# Cell 2: Install
install_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 고품질 시각화를 위한 패키지 설치\n",
        "!pip install seaborn plotly"
    ]
}
cells_to_add.append(install_cell)

# Cell 3: Data prep & plotting
code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import plotly.express as px\n",
        "from rdkit import Chem\n",
        "from rdkit.Chem import AllChem\n",
        "from sklearn.manifold import TSNE\n",
        "import os\n",
        "\n",
        "# --- 1. Styles and Paths ---\n",
        "sns.set_theme(style=\"ticks\", context=\"paper\", font_scale=1.2)\n",
        "plt.rcParams[\"figure.dpi\"] = 300\n",
        "RESULTS_CHEM_SPACE = '/content/drive/MyDrive/Attentive_FP/Results/Corr/ChemicalSpace/'\n",
        "os.makedirs(RESULTS_CHEM_SPACE, exist_ok=True)\n",
        "\n",
        "# --- 2. Data Preparation ---\n",
        "print(\"🔬 데이터 결합 및 Morgan Fingerprint 추출 중...\")\n",
        "train_df['Dataset'] = 'Train'\n",
        "test_df['Dataset'] = 'Test'\n",
        "combined_df = pd.concat([train_df, test_df], ignore_index=True)\n",
        "combined_df['GHS_Label'] = combined_df['label'].map({1: 'Corrosive (Toxic)', 0: 'Negative (Non-Toxic)'})\n",
        "\n",
        "def get_morgan_fp(smiles, radius=2, nBits=2048):\n",
        "    mol = Chem.MolFromSmiles(smiles)\n",
        "    if mol:\n",
        "        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)\n",
        "        return np.array(fp)\n",
        "    return np.zeros((nBits,))\n",
        "\n",
        "# Extract fingerprints\n",
        "fps = np.array([get_morgan_fp(smi) for smi in combined_df['smiles']])\n",
        "print(f\"✅ 총 {len(fps)}개의 분자에서 Fingerprint 추출 완료!\")\n",
        "\n",
        "# --- 3. t-SNE Calculation ---\n",
        "print(\"🔄 t-SNE 2D 및 3D 좌표 계산 중... (수 분이 소요될 수 있습니다)\")\n",
        "# 2D t-SNE\n",
        "tsne_2d = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')\n",
        "fps_tsne_2d = tsne_2d.fit_transform(fps)\n",
        "combined_df['t-SNE_1'] = fps_tsne_2d[:, 0]\n",
        "combined_df['t-SNE_2'] = fps_tsne_2d[:, 1]\n",
        "\n",
        "# 3D t-SNE\n",
        "tsne_3d = TSNE(n_components=3, random_state=42, init='pca', learning_rate='auto')\n",
        "fps_tsne_3d = tsne_3d.fit_transform(fps)\n",
        "combined_df['t-SNE_3d_1'] = fps_tsne_3d[:, 0]\n",
        "combined_df['t-SNE_3d_2'] = fps_tsne_3d[:, 1]\n",
        "combined_df['t-SNE_3d_3'] = fps_tsne_3d[:, 2]\n",
        "print(\"✅ t-SNE 계산 완료!\")\n",
        "\n",
        "# --- 4. 2D t-SNE 시각화 (Seaborn) ---\n",
        "plt.figure(figsize=(10, 8))\n",
        "palette = {'Corrosive (Toxic)': '#e74c3c', 'Negative (Non-Toxic)': '#3498db'}\n",
        "\n",
        "sns.scatterplot(\n",
        "    data=combined_df, x='t-SNE_1', y='t-SNE_2', \n",
        "    hue='GHS_Label', style='Dataset',\n",
        "    palette=palette, alpha=0.7, s=60, edgecolor='white'\n",
        ")\n",
        "plt.title('t-SNE Projection of Chemical Space (2D)', fontweight='bold', pad=15)\n",
        "plt.xlabel('t-SNE Dimension 1')\n",
        "plt.ylabel('t-SNE Dimension 2')\n",
        "plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n",
        "plt.tight_layout()\n",
        "\n",
        "save_path_tsne2d = os.path.join(RESULTS_CHEM_SPACE, 'tSNE_2D_Plot.png')\n",
        "plt.savefig(save_path_tsne2d, dpi=300, bbox_inches='tight')\n",
        "print(f\"✅ 2D t-SNE 플롯 저장 완료: {save_path_tsne2d}\")\n",
        "plt.show()\n",
        "\n",
        "# --- 5. 3D t-SNE 시각화 (Plotly) ---\n",
        "print(\"🔄 3D t-SNE 그래프를 생성합니다...\")\n",
        "fig_3d_tsne = px.scatter_3d(\n",
        "    combined_df, x='t-SNE_3d_1', y='t-SNE_3d_2', z='t-SNE_3d_3',\n",
        "    color='GHS_Label', symbol='Dataset',\n",
        "    color_discrete_map=palette, opacity=0.7,\n",
        "    title='t-SNE Projection of Chemical Space (3D)'\n",
        ")\n",
        "fig_3d_tsne.update_traces(marker=dict(size=4, line=dict(width=1, color='White')))\n",
        "fig_3d_tsne.update_layout(margin=dict(l=0, r=0, b=0, t=40))\n",
        "\n",
        "# HTML로 저장\n",
        "save_path_tsne3d = os.path.join(RESULTS_CHEM_SPACE, 'tSNE_3D_Interactive.html')\n",
        "fig_3d_tsne.write_html(save_path_tsne3d)\n",
        "print(f\"✅ 3D t-SNE (인터랙티브) 저장 완료: {save_path_tsne3d}\")\n",
        "fig_3d_tsne.show()"
    ]
}
cells_to_add.append(code_cell)

nb['cells'].extend(cells_to_add)

with open('Corr_SRP_ChemSpace.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("[OK] Notebook updated with clean t-SNE code")
