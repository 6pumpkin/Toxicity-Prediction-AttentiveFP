#!/usr/bin/env python3
"""
Corr_SRP.ipynb (하이브리드 모델)에서 머신러닝(RDKit 디스크립터) 부분을 제거하고
Attention(GNN) 모델만 남긴 새 노트북 Corr_SRP_AttentionOnly.ipynb를 생성하는 스크립트.

변경 사항 요약:
- Cell 8: RDKIT_DESC_NAMES, calculate_rdkit_descriptors, HybridModel 제거. GNN_Extractor만 유지.
- Cell 10: smiles_to_graph_hybrid -> smiles_to_graph (scaler/descriptor 코드 제거)
- Cell 12: HybridModel -> GNN_Extractor, scaler 관련 코드 제거
- Cell 17: visualize_attention/find_and_visualize 함수에서 scaler 파라미터 제거
- Cell 22: 데이터 로딩에서 descriptor_scaler 관련 코드 전체 제거
- Cell 27: HybridModel -> GNN_Extractor 직접 사용
- Cell 31: scaler 파라미터 제거
- Cell 34: SHAP 분석을 어텐션 기반 해석으로 대체
"""

import json, copy

with open('Corr_SRP.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_nb = copy.deepcopy(nb)
cells = new_nb['cells']

# 모든 셀의 출력(outputs) 초기화
for cell in cells:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

# ============================================================
# Cell 8: 모델 정의 - HybridModel 제거, GNN_Extractor만 유지
# ============================================================
src8 = ''.join(cells[8]['source'])

# RDKIT_DESC_NAMES ~ calculate_rdkit_descriptors 함수 제거
cut_start = src8.find("# RDKit")
if cut_start < 0:
    cut_start = src8.find("RDKIT_DESC_NAMES")

cut_end = src8.find("# =================================================================\n# 분자 구조를 위한")
if cut_end < 0:
    cut_end = src8.find("def one_of_k_encoding(")

if cut_start >= 0 and cut_end >= 0:
    src8 = src8[:cut_start] + src8[cut_end:]

# HybridModel 클래스 전체 제거
hybrid_start = src8.find("\nclass HybridModel(nn.Module):")
if hybrid_start < 0:
    hybrid_start = src8.find("class HybridModel(nn.Module):")

if hybrid_start >= 0:
    src8 = src8[:hybrid_start].rstrip() + "\n"

cells[8]['source'] = [src8]

# ============================================================
# Cell 10: smiles_to_graph_hybrid -> smiles_to_graph (scaler 제거)
# ============================================================
src10 = ''.join(cells[10]['source'])

# smiles_to_graph_hybrid 함수 전체를 새 함수로 교체
func_start = src10.find("def smiles_to_graph_hybrid(")
if func_start >= 0:
    new_func = '''def smiles_to_graph(smiles_string):
    """
    [Attention Only] SMILES 문자열로부터 GNN 입력용 Data 객체를 생성합니다.
    RDKit 디스크립터 관련 코드가 제거되었습니다.
    """
    try:
        # 1. 분자 객체 생성
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return None

        # 2. GNN을 위한 원자 특징(x) 생성
        atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_feats, dtype=torch.float)

        # 3. GNN을 위한 엣지 특징(edge_index, edge_attr) 생성
        if mol.GetNumBonds() > 0:
            bond_feats = [bond_features(bond) for bond in mol.GetBonds()]
            edge_indices = []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_indices.extend([(i, j), (j, i)])
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(bond_feats + bond_feats, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, num_bond_features()), dtype=torch.float)

        # 4. Data 객체 생성 및 반환
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles_string)
        return data

    except Exception as e:
        return None
'''
    src10 = src10[:func_start] + new_func

cells[10]['source'] = [src10]

# ============================================================
# Cell 12: Scaffold 교차검증 함수 - HybridModel -> GNN_Extractor
# ============================================================
src12 = ''.join(cells[12]['source'])

# run_scaffold_cross_validation 함수 내부의 모델 생성 부분 수정
# 기존: GNN_Extractor + HybridModel  ->  GNN_Extractor만
old_model_block_12 = """    # RDKit 디스크립터 개수는 미리 계산해 둡니다.
    num_descriptors = len(RDKIT_DESC_NAMES)"""
new_model_block_12 = """    # [Attention Only] RDKit 디스크립터 관련 코드 제거됨"""
src12 = src12.replace(old_model_block_12, new_model_block_12)

# HybridModel 래핑 부분 제거
old_hybrid_12 = """                    # 2. HybridModel로 최종 모델을 감싸서 초기화합니다.
                    model = HybridModel(
                        gnn_model=gnn_part,
                        num_descriptors=num_descriptors,
                        gnn_feature_dim=hidden
                    ).to(device)
                    # ---------------------------"""
new_hybrid_12 = """                    # [Attention Only] GNN_Extractor를 직접 모델로 사용합니다.
                    model = gnn_part
                    # ---------------------------"""
src12 = src12.replace(old_hybrid_12, new_hybrid_12)

cells[12]['source'] = [src12]

# ============================================================
# Cell 17: 시각화 함수 - scaler 파라미터 제거
# ============================================================
src17 = ''.join(cells[17]['source'])

# visualize_attention 함수 시그니처에서 scaler 제거
src17 = src17.replace(
    "def visualize_attention(model, smiles_string, train_fps, threshold_S, fp_function, scaler, save_path_prefix, threshold=None):",
    "def visualize_attention(model, smiles_string, train_fps, threshold_S, fp_function, save_path_prefix, threshold=None):"
)

# smiles_to_graph_hybrid(smiles_string, scaler) -> smiles_to_graph(smiles_string)
src17 = src17.replace("smiles_to_graph_hybrid(smiles_string, scaler)", "smiles_to_graph(smiles_string)")

# find_and_visualize_extreme_predictions 함수 시그니처에서 scaler 제거
src17 = src17.replace(
    "def find_and_visualize_extreme_predictions(model, dataset, train_fps, threshold_S, fp_function, scaler, top_n=10):",
    "def find_and_visualize_extreme_predictions(model, dataset, train_fps, threshold_S, fp_function, top_n=10):"
)

# visualize_attention 호출에서 scaler 제거
src17 = src17.replace(
    "model, item['smiles'], train_fps, threshold_S, fp_function, scaler, save_path_prefix=save_prefix",
    "model, item['smiles'], train_fps, threshold_S, fp_function, save_path_prefix=save_prefix"
)

cells[17]['source'] = [src17]

# ============================================================
# Cell 22: 데이터 로딩 - descriptor_scaler 관련 코드 제거
# ============================================================
new_cell22_source = '''# (필요한 import문들이 있다고 가정: pandas, numpy, torch, tqdm, sklearn 등)
# (smiles_to_graph 함수가 미리 정의되어 있다고 가정)

try:
    # --- 1. 데이터 로딩 및 기본 전처리 ---
    print("1. 데이터를 로딩하고 기본 전처리를 수행합니다...")
    file_path = '/content/drive/MyDrive/Attentive_FP/Corr.xlsx'
    full_dataset = pd.read_excel(file_path)
    corr_df = full_dataset[full_dataset['GHS'].isin(['Corr', 'Neg'])].copy()
    corr_df['label'] = corr_df['GHS'].apply(lambda x: 1 if x == 'Corr' else 0)
    corr_df['smiles'] = full_dataset.loc[corr_df.index, 'SMILES'].copy()
    df = corr_df[['smiles', 'label']].copy().dropna().reset_index(drop=True)
    print(f"  -> 총 {len(df)}개의 분자 데이터를 로드했습니다.")

    # --- 2. DataFrame을 훈련/테스트 세트로 먼저 분할 ---
    print("\\n2. DataFrame을 훈련/테스트 세트로 분할합니다...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    print(f"  -> 훈련 세트: {len(train_df)}개, 테스트 세트: {len(test_df)}개")

    # --- 3. [Attention Only] 디스크립터 스케일러 단계 생략 ---
    print("\\n3. [Attention Only] 디스크립터 스케일러 단계를 생략합니다.")

    # --- 4. PyG 데이터셋 생성 (GNN 그래프만 사용) ---
    print("\\n4. PyTorch Geometric 데이터셋을 생성합니다...")
    # 훈련 데이터셋 생성
    train_dataset = []
    for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc="훈련 세트 변환 중"):
        data_obj = smiles_to_graph(row['smiles'])
        if data_obj is not None:
            data_obj.y = torch.tensor([row['label']], dtype=torch.float)
            train_dataset.append(data_obj)

    # 테스트 데이터셋 생성
    test_dataset = []
    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="테스트 세트 변환 중"):
        data_obj = smiles_to_graph(row['smiles'])
        if data_obj is not None:
            data_obj.y = torch.tensor([row['label']], dtype=torch.float)
            test_dataset.append(data_obj)

    print(f"  -> ✅ 변환 완료: 훈련 데이터 {len(train_dataset)}개, 테스트 데이터 {len(test_dataset)}개")

    # --- 5. 후속 분석을 위한 변수 생성 (AD 계산 등) ---
    print("\\n5. 후속 분석을 위한 변수를 생성합니다...")
    train_smiles_list = [data.smiles for data in train_dataset]
    if train_smiles_list:
        ad_threshold, train_fingerprints = calculate_similarity_threshold(
            train_smiles_list,
            fp_function=smiles_to_maccs_fp,
            Z=1.0
        )
        print("✅ AD 계산 및 지문 생성 완료!")
        print(f"   -> 반환된 임계값: {ad_threshold:.4f}")
        print(f"   -> 반환된 지문 수: {len(train_fingerprints)}")
    else:
        print("  -> ⚠️ 훈련 데이터가 없어 AD 계산을 건너뜁니다.")
        ad_threshold, train_fingerprints = 0.0, []

except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
    print("프로그램 실행을 중단합니다. 파일 경로를 확인하고 다시 시도해 주세요.")
    raise
except Exception as e:
    print("\\n" + "="*50)
    print(f"❌ 데이터 처리 중 예측하지 못한 오류 발생: {e}")
    traceback.print_exc()
    print("="*50)
'''
cells[22]['source'] = [new_cell22_source]

# ============================================================
# Cell 27: 최종 학습 - GNN_Extractor 직접 사용
# ============================================================
new_cell27_source = '''# ===================================================================
# 3.4: 최종 모델 학습 및 평가 [Attention Only]
# ===================================================================
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(f"이번 실행의 고유 ID: {run_timestamp}")
SUBFOLDER_FR = '/content/drive/MyDrive/Attentive_FP/Results/Corr/Final_result/'

print("\\n--- 최종 모델 학습 시작 ---")

# 1. 학습 환경 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"최종 학습에 사용할 장치: {device}")

# 2. [Attention Only] GNN_Extractor를 직접 최종 모델로 사용
gnn_hidden_channels = best_params['hidden']
final_model = GNN_Extractor(
    hidden_channels=gnn_hidden_channels,
    out_channels=1,
    dropout=best_params['dropout']
).to(device)

# 3. 옵티마이저에 모델의 모든 파라미터를 전달
optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
criterion = nn.BCEWithLogitsLoss()

# 4. 데이터 로더 준비 (훈련/검증/테스트)
train_subset, val_subset = train_test_split(train_dataset, test_size=0.2, random_state=42)
final_train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
final_val_loader = DataLoader(val_subset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# 5. 조기 중지(Early Stopping) 기능으로 모델 학습 실행
final_model, history = train_with_early_stopping(
    model=final_model,
    train_loader=final_train_loader,
    val_loader=final_val_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    epochs=100,
    patience=10
)

# 6. 학습 과정 시각화
plot_learning_curves(history)

# 7. 최종 모델 성능 평가
final_metrics, y_true_final, y_pred_final, y_pred_proba_final = evaluate(final_model, test_loader, criterion, device)

# 저장할 파일 경로 지정
final_results_path = os.path.join(SUBFOLDER_FR, f"final_result_{run_timestamp}.txt")

with open(final_results_path, 'w') as f:
    with redirect_stdout(f):
        print("✅ 최종 테스트 데이터셋 평가 지표 (외부 검증 결과):")
        print("-" * 50)
        for name, value in final_metrics.items():
            print(f"  {name:<10s}: {format_metric(value)}%")
        print("-" * 50)

print(f"✅ 최종 평가 결과가 '{final_results_path}'에 저장되었습니다.")

# 평가 지표 출력
print(f"\\n✅ 최종 테스트 데이터셋 평가 지표 (외부 검증 결과):")
print("-" * 50)
for name, value in final_metrics.items():
    print(f"  {name:<10s}: {value:.2f}%")
print("-" * 50)

# 혼동 행렬 시각화
plot_confusion_matrix(y_true_final, y_pred_final)

# ROC Curve 시각화 함수 호출
plot_roc_curve(y_true_final, y_pred_proba_final)
'''
cells[27]['source'] = [new_cell27_source]

# ============================================================
# Cell 29: AD 분석 - final_hybrid_model -> final_model
# ============================================================
src29 = ''.join(cells[29]['source'])
src29 = src29.replace('final_hybrid_model', 'final_model')
cells[29]['source'] = [src29]

# ============================================================
# Cell 31: 시각화 호출 - scaler 파라미터 제거
# ============================================================
new_cell31_source = '''find_and_visualize_extreme_predictions(
    model=final_model,
    dataset=test_dataset,
    train_fps=train_fingerprints,
    threshold_S=ad_threshold,
    fp_function=smiles_to_maccs_fp
)
'''
cells[31]['source'] = [new_cell31_source]

# ============================================================
# Cell 33: Structure alerts - final_hybrid_model -> final_model
# ============================================================
src33 = ''.join(cells[33]['source'])
src33 = src33.replace('final_hybrid_model', 'final_model')
cells[33]['source'] = [src33]

# ============================================================
# Cell 34: SHAP 분석 - RDKit 디스크립터 기반 -> 어텐션 해석으로 대체
# ============================================================
new_cell34_source = '''# ===================================================================
# [Attention Only] 모델 해석
# ===================================================================
# 하이브리드 모델에서는 RDKit 디스크립터를 기반으로 SHAP 분석을 수행했습니다.
# Attention Only 모델에서는 어텐션 가중치(attention weights)가
# 각 원자/결합의 중요도를 직접 보여주므로, 별도의 SHAP 분석 대신
# 위의 시각화 함수(find_and_visualize_extreme_predictions)의
# 어텐션 맵을 통해 모델의 해석 가능성을 확인할 수 있습니다.

print("=" * 50)
print("✅ Attention Only 모델 해석 안내")
print("=" * 50)
print()
print("이 모델은 Graph Attention Network(GAT) 기반의 AttentiveFP 구조를 사용합니다.")
print("모델의 해석은 다음 두 가지 방법으로 수행됩니다:")
print()
print("1. 어텐션 맵 시각화 (위의 3.6 섹션)")
print("   - 각 원자에 대한 어텐션 가중치를 색상으로 표현")
print("   - 빨간색: 높은 기여도 / 파란색: 낮은 기여도")
print()
print("2. 구조 경고 분석 (위의 3.7 섹션)")
print("   - Morgan Fingerprint 기반 독성 관련 구조적 특징 분석")
print()
print("✅ 별도의 SHAP 분석은 필요하지 않습니다.")
'''
cells[34]['source'] = [new_cell34_source]

# ============================================================
# 나머지 셀들에서 final_hybrid_model -> final_model 일괄 치환
# ============================================================
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'final_hybrid_model' in src:
            src = src.replace('final_hybrid_model', 'final_model')
            cells[i]['source'] = [src]

# ============================================================
# widgets 메타데이터 제거 (GitHub 렌더링 호환성)
# ============================================================
metadata = new_nb.get('metadata', {})
if 'widgets' in metadata:
    del metadata['widgets']

# ============================================================
# 노트북 저장
# ============================================================
output_path = 'Corr_SRP_AttentionOnly.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, ensure_ascii=False, indent=1)

print(f"[OK] Attention Only notebook saved: '{output_path}'")
