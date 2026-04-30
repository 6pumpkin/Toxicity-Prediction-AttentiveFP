import json
import copy

# 1. Load the base notebook
with open('Corr_SRP_ChemSpace.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 2. Duplicate it
new_nb = copy.deepcopy(nb)
cells = new_nb['cells']

# Helper to clear outputs
for cell in cells:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

# 3. Modification Rules
for i, cell in enumerate(cells):
    if cell['cell_type'] != 'code':
        continue
    
    src = ''.join(cell['source'])
    
    # Cell 2: Install pip (Add optuna)
    if '!pip install' in src and 'torch==' in src:
        new_src = []
        for line in cell['source']:
            if '!pip install torch==' in line:
                # Ensure torch 2.3.0 and torchvision 0.18.0
                if 'torchvision' not in line:
                    line = line.replace('!pip install torch==2.3.0', '!pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0')
            elif '!pip install' in line and 'shap' in line:
                if 'optuna' not in line:
                    line = line.replace('\n', '') + ' optuna\n'
            new_src.append(line)
        cell['source'] = new_src

    # Cell 10: train_with_early_stopping (Add LR Scheduler)
    if 'def train_with_early_stopping' in src:
        new_func = """def train_with_early_stopping(model, train_loader, val_loader, test_loader, optimizer, criterion, device, epochs=100, patience=20):
    \"\"\"조기 중지(Early Stopping)와 학습률 스케줄러(ReduceLROnPlateau)를 포함합니다.\"\"\"
    # --- 스케줄러 추가 ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'test_auc': [],
               'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        # 1. 모델 학습
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 2. 성능 검증 (Validation)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
        val_loss, val_auc, val_acc = val_metrics['loss'], val_metrics['AUC'], val_metrics['ACC']
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_acc'].append(val_acc)

        # 3. 테스트
        test_metrics, _, _, _ = evaluate(model, test_loader, criterion, device)
        test_auc = test_metrics['AUC']
        history['test_auc'].append(test_auc)
        
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:02d} [LR: {current_lr:.6f}]: Train Loss: {format_metric(train_loss)}, Train Acc: {format_metric(train_acc)}% | "
              f"Val Loss: {format_metric(val_loss)}, Val Acc: {format_metric(val_acc)}%, Val AUC: {format_metric(val_auc)}% | "
              f"Test AUC: {format_metric(test_auc)}%")

        # --- 스케줄러 스텝 (검증 손실 기반) ---
        scheduler.step(val_loss)

        # --- 조기 중지 조건 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\\n>> {patience} 에포크 동안 검증 손실이 개선되지 않아 조기 중지합니다.")
            break

    if best_model_state:
        print(f"\\n가장 좋았던 모델 (Val Loss: {best_val_loss:.4f})의 상태를 불러옵니다.")
        model.load_state_dict(best_model_state)

    return model, history
"""
        # Find start of train_with_early_stopping and replace
        func_start = src.find('def train_with_early_stopping(')
        if func_start >= 0:
            cell['source'] = [src[:func_start] + new_func]

    # Cell 12: run_scaffold_cross_validation (Replace with Optuna)
    if 'def run_scaffold_cross_validation' in src:
        optuna_func = """import optuna

def run_scaffold_cross_validation(train_dataset, n_folds=5, n_trials=20):
    \"\"\"
    Optuna를 활용한 스캐폴드 교차검증 기반 하이퍼파라미터 최적화
    \"\"\"
    scaffold_folds = scaffold_split(train_dataset, n_folds=n_folds)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\\n교차검증에 사용할 장치: {device}")
    
    def objective(trial):
        # 1. 넓은 탐색 공간 (Search Space) 정의
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        hidden = trial.suggest_categorical("hidden_channels", [16, 24, 32, 64])
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        
        fold_aucs = []
        for i in range(n_folds):
            val_idx = scaffold_folds[i]
            train_idx = [idx for j, fold in enumerate(scaffold_folds) if i != j for idx in fold]

            train_fold_dataset = [train_dataset[k] for k in train_idx]
            val_fold_dataset = [train_dataset[k] for k in val_idx]

            train_loader = DataLoader(train_fold_dataset, batch_size=64, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_fold_dataset, batch_size=64, drop_last=False)

            gnn_part = GNN_Extractor(
                hidden_channels=hidden,
                out_channels=1,
                dropout=dropout
            ).to(device)

            optimizer = optim.Adam(gnn_part.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.BCEWithLogitsLoss()
            
            # 스케줄러 적용 (빠른 탐색을 위해 patience를 3으로 설정)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            
            for epoch in range(35): # 빠른 평가를 위해 에포크 35회 제한
                train_loss, _ = train(gnn_part, train_loader, optimizer, criterion, device)
                val_metrics, _, _, _ = evaluate(gnn_part, val_loader, criterion, device)
                scheduler.step(val_metrics['loss'])

            val_metrics, _, _, _ = evaluate(gnn_part, val_loader, criterion, device)
            fold_aucs.append(val_metrics['AUC'])
            
        return np.mean(fold_aucs)
        
    print("\\n--- Optuna 하이퍼파라미터 탐색 시작 ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING) # 로그 축소
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print("\\n--- 스캐폴드 교차검증 종료 ---")
    best_params = study.best_params
    print(f"최적 하이퍼파라미터: {best_params} (평균 AUC: {study.best_value:.2f}%)")
    
    return {
        'lr': best_params['lr'],
        'hidden': best_params['hidden_channels'],
        'dropout': best_params['dropout'],
        'weight_decay': best_params['weight_decay']
    }
"""
        func_start = src.find('def run_scaffold_cross_validation(')
        if func_start >= 0:
            cell['source'] = [src[:func_start] + optuna_func]

    # Cell 25: Call Optuna instead of grid search
    if 'param_grid' in src and 'best_params' in src and 'run_scaffold_cross_validation' in src:
        new_call = """# Optuna를 호출하여 최적의 파라미터를 탐색합니다.
print("\\n--- 스캐폴드 기반 Optuna 최적화 (5-Fold, 20 Trials) ---")
# n_folds=5, n_trials=20으로 설정하여 시간과 성능의 밸런스를 맞춥니다.
best_params = run_scaffold_cross_validation(train_dataset, n_folds=5, n_trials=20)
"""
        cell['source'] = [new_call]

    # Cell 27: Final train (Add weight_decay and update patience to 20)
    if 'optimizer = optim.Adam' in src and 'final_model' in src:
        new_src = src.replace(
            "optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])",
            "optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params.get('weight_decay', 0.0))"
        )
        new_src = new_src.replace(
            "patience=10",
            "patience=20"
        )
        cell['source'] = [new_src]

# 4. Save to new file
out_file = 'Corr_SRP_Optuna.ipynb'
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, ensure_ascii=False, indent=1)

print(f"[OK] Successfully applied Optuna, LR Schedulers, and Expanded Search Space to {out_file}")
