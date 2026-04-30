import json
import copy

# 1. 원본 노트북 로드
with open('Corr_SRP.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 2. 복사본 생성
new_nb = copy.deepcopy(nb)

# 3. 추가할 셀들 정의
cells_to_add = []

# Cell 1: Markdown
markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 3.8 데이터 분포 및 화학 공간 분석 (Chemical Space Analysis)\n",
        "t-SNE와 UMAP을 활용하여 훈련 및 테스트 데이터셋의 분자들의 구조적 분포를 2차원 및 3차원 화학 공간에 시각화합니다.\n",
        "\n",
        "**t-SNE (t-Distributed Stochastic Neighbor Embedding)**:\n",
        "- 고차원 데이터(분자 지문)를 2~3차원으로 축소할 때, **가까이 있는 데이터(비슷한 분자)는 계속 가깝게 유지**하도록 맵핑하는 데 특화된 알고리즘입니다.\n",
        "- 지역적(Local) 군집을 아주 잘 보여주어, 독성을 띠는 분자들이 특정 클러스터로 모이는지 확인하기 좋습니다.\n",
        "\n",
        "**UMAP (Uniform Manifold Approximation and Projection)**:\n",
        "- t-SNE의 단점인 '느린 속도'와 '전체적인 구조(Global structure) 왜곡'을 보완한 최신 차원 축소 기법입니다.\n",
        "- 지역적인 군집뿐만 아니라, 클러스터 간의 거리(전체적인 화학 공간의 형태)도 더 의미 있게 보존합니다. 최근 Cheminformatics 논문에서 매우 활발히 쓰입니다."
    ]
}
cells_to_add.append(markdown_cell)

# Cell 2: Install UMAP
install_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# UMAP 및 고품질 시각화를 위한 패키지 설치\n",
        "!pip install umap-learn seaborn plotly"
    ]
}
cells_to_add.append(install_cell)

# Cell 3: Data preparation
data_prep_cell = {
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
        "import umap.umap_ as umap\n",
        "import os\n",
        "\n",
        "# 고품질 그래프 스타일 설정\n",
        "sns.set_theme(style=\"ticks\", context=\"paper\", font_scale=1.2)\n",
        "plt.rcParams[\"figure.dpi\"] = 300\n",
        "\n",
        "# 결과를 저장할 폴더 생성\n",
        "RESULTS_CHEM_SPACE = '/content/drive/MyDrive/Attentive_FP/Results/Corr/ChemicalSpace/'\n",
        "os.makedirs(RESULTS_CHEM_SPACE, exist_ok=True)\n",
        "\n",
        "print(\"🔬 데이터 결합 및 Morgan Fingerprint 추출 중...\")\n",
        "\n",
        "# 훈련 및 테스트 데이터 병합 (시각화용)\n",
        "train_df['Dataset'] = 'Train'\n",
        "test_df['Dataset'] = 'Test'\n",
        "combined_df = pd.concat([train_df, test_df], ignore_index=True)\n",
        "\n",
        "# 라벨 매핑 (시각화 시 명확성을 위해)\n",
        "combined_df['GHS_Label'] = combined_df['label'].map({1: 'Corrosive (Toxic)', 0: 'Negative (Non-Toxic)'})\n",
        "\n",
        "# Morgan Fingerprint 계산 (반경 2, 2048 비트 - 표준 논문 설정)\n",
        "def get_morgan_fp(smiles, radius=2, nBits=2048):\n",
        "    mol = Chem.MolFromSmiles(smiles)\n",
        "    if mol:\n",
        "        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)\n",
        "        return np.array(fp)\n",
        "    return np.zeros((nBits,))\n",
        "\n",
        "# 지문 추출\n",
        "fps = np.array([get_morgan_fp(smi) for smi in combined_df['smiles']])\n",
        "print(f\"✅ 총 {len(fps)}개의 분자에서 Fingerprint 추출 완료!\")"
    ]
}
cells_to_add.append(data_prep_cell)

# Cell 4: t-SNE 2D/3D
tsne_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
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
        "# --- 2D t-SNE 시각화 (Seaborn) ---\n",
        "plt.figure(figsize=(10, 8))\n",
        "palette = {'Corrosive (Toxic)': '#e74c3c', 'Negative (Non-Toxic)': '#3498db'}\n",
        "\n",
        "sns.scatterplot(\n",
        "    data=combined_df, x='t-SNE_1', y='t-SNE_2',\n",
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
        "# --- 3D t-SNE 시각화 (Plotly - 인터랙티브) ---\n",
        "fig_3d_tsne = px.scatter_3d(\n",
        "    combined_df, x='t-SNE_3d_1', y='t-SNE_3d_2', z='t-SNE_3d_3',\n",
        "    color='GHS_Label', symbol='Dataset',\n",
        "    color_discrete_map=palette, opacity=0.7,\n",
        "    title='t-SNE Projection of Chemical Space (3D)'\n",
        ")\n",
        "fig_3d_tsne.update_traces(marker=dict(size=4, line=dict(width=1, color='White')))\n",
        "fig_3d_tsne.update_layout(margin=dict(l=0, r=0, b=0, t=40))\n",
        "\n",
        "# HTML로 인터랙티브하게 저장\n",
        "save_path_tsne3d = os.path.join(RESULTS_CHEM_SPACE, 'tSNE_3D_Interactive.html')\n",
        "fig_3d_tsne.write_html(save_path_tsne3d)\n",
        "print(f\"✅ 3D t-SNE (인터랙티브) 저장 완료: {save_path_tsne3d}\")\n",
        "fig_3d_tsne.show()"
    ]
}
cells_to_add.append(tsne_cell)

# Cell 5: UMAP 2D/3D
umap_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"🔄 UMAP 2D 및 3D 좌표 계산 중...\")\n",
        "# 2D UMAP\n",
        "reducer_2d = umap.UMAP(n_components=2, random_state=42, metric='jaccard')\n",
        "fps_umap_2d = reducer_2d.fit_transform(fps)\n",
        "combined_df['UMAP_1'] = fps_umap_2d[:, 0]\n",
        "combined_df['UMAP_2'] = fps_umap_2d[:, 1]\n",
        "\n",
        "# 3D UMAP\n",
        "reducer_3d = umap.UMAP(n_components=3, random_state=42, metric='jaccard')\n",
        "fps_umap_3d = reducer_3d.fit_transform(fps)\n",
        "combined_df['UMAP_3d_1'] = fps_umap_3d[:, 0]\n",
        "combined_df['UMAP_3d_2'] = fps_umap_3d[:, 1]\n",
        "combined_df['UMAP_3d_3'] = fps_umap_3d[:, 2]\n",
        "print(\"✅ UMAP 계산 완료!\")\n",
        "\n",
        "# --- 2D UMAP 시각화 (Seaborn) ---\n",
        "plt.figure(figsize=(10, 8))\n",
        "sns.scatterplot(\n",
        "    data=combined_df, x='UMAP_1', y='UMAP_2',\n",
        "    hue='GHS_Label', style='Dataset',\n",
        "    palette=palette, alpha=0.7, s=60, edgecolor='white'\n",
        ")\n",
        "plt.title('UMAP Projection of Chemical Space (2D)', fontweight='bold', pad=15)\n",
        "plt.xlabel('UMAP Dimension 1')\n",
        "plt.ylabel('UMAP Dimension 2')\n",
        "plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n",
        "plt.tight_layout()\n",
        "\n",
        "save_path_umap2d = os.path.join(RESULTS_CHEM_SPACE, 'UMAP_2D_Plot.png')\n",
        "plt.savefig(save_path_umap2d, dpi=300, bbox_inches='tight')\n",
        "print(f\"✅ 2D UMAP 플롯 저장 완료: {save_path_umap2d}\")\n",
        "plt.show()\n",
        "\n",
        "# --- 3D UMAP 시각화 (Plotly - 인터랙티브) ---\n",
        "fig_3d_umap = px.scatter_3d(\n",
        "    combined_df, x='UMAP_3d_1', y='UMAP_3d_2', z='UMAP_3d_3',\n",
        "    color='GHS_Label', symbol='Dataset',\n",
        "    color_discrete_map=palette, opacity=0.7,\n",
        "    title='UMAP Projection of Chemical Space (3D)'\n",
        ")\n",
        "fig_3d_umap.update_traces(marker=dict(size=4, line=dict(width=1, color='White')))\n",
        "fig_3d_umap.update_layout(margin=dict(l=0, r=0, b=0, t=40))\n",
        "\n",
        "save_path_umap3d = os.path.join(RESULTS_CHEM_SPACE, 'UMAP_3D_Interactive.html')\n",
        "fig_3d_umap.write_html(save_path_umap3d)\n",
        "print(f\"✅ 3D UMAP (인터랙티브) 저장 완료: {save_path_umap3d}\")\n",
        "fig_3d_umap.show()"
    ]
}
cells_to_add.append(umap_cell)

# 4. 기존 셀 리스트 뒤에 추가
new_nb['cells'].extend(cells_to_add)

# 5. 위젯 메타데이터(GitHub 렌더링 에러 방지) 제거
if 'metadata' in new_nb and 'widgets' in new_nb['metadata']:
    del new_nb['metadata']['widgets']

# 6. 새 이름으로 저장
out_file = 'Corr_SRP_ChemSpace.ipynb'
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, ensure_ascii=False, indent=1)

print(f"[OK] Successfully appended cells and saved to {out_file}")
