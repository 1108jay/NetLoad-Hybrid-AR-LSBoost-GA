%% [실증분석] Hybrid AR-Lasso-LSBoost 융합 모델 (고도화 1단계)
% 적용 사항: 시계열 순차 분할(Chronological Split) 및 이동평균 파생변수 추가

clc; clear; close all;
rng(20260409);

%% 1. 데이터 로드 및 파생 변수(Feature Engineering) 고도화
fprintf('마스터 테이블 로딩 및 시계열 전처리 중...\n');
Tbl = readtable('Final_Master_Table_for_MATLAB.csv', 'VariableNamingRule', 'preserve');

% 신호처리 융합: 푸리에 변환 기반 주기성 변수
Hour = hour(Tbl.Datetime);
Tbl.SinHour = sin(2*pi*Hour/24);
Tbl.CosHour = cos(2*pi*Hour/24);

% 추가 고도화 (이동평균 변수): 날씨의 순간적 튐 현상(Noise)을 통제
% 최근 3시간 기온 평균, 최근 3시간 일사량 평균 파생변수 생성
Tbl.Temp_MA3 = movmean(Tbl.('기온(°C)'), [2 0]); 
Tbl.Insol_MA3 = movmean(Tbl.('일사(MJ/m2)'), [2 0]);

%% 2. 과적합 방지를 위한 시계열 순차 분할 (Chronological Split)
% 무작위 분할(Holdout) 폐기 -> 2022~2024년 Train, 2025년 Test
train_idx = year(Tbl.Datetime) < 2025;
test_idx  = year(Tbl.Datetime) == 2025;

TblTrain = Tbl(train_idx, :);
TblTest  = Tbl(test_idx, :);

fprintf('Train 데이터 (2022~2024): %d 건\n', sum(train_idx));
fprintf('Test 데이터 (2025년): %d 건\n', sum(test_idx));

%% 3. Phase 1: 전통 계량 모형 (AR Baseline)
fprintf('\n[Phase 1] AR 베이스라인 모형 적합 중...\n');
formula_AR = 'Net_Load ~ Net_Load_Lag24 + Net_Load_Lag168';
mdl_AR = fitlm(TblTrain, formula_AR);

AR_Pred_Train = predict(mdl_AR, TblTrain);
AR_Pred_Test  = predict(mdl_AR, TblTest);

Res_Train = TblTrain.Net_Load - AR_Pred_Train;
Res_Test  = TblTest.Net_Load  - AR_Pred_Test;

%% 4. Phase 2: Lasso 기반 Feature Selection
fprintf('[Phase 2] Lasso 정규화를 통한 최적 기상/파생 변수 추출 중...\n');

% 추가된 이동평균 변수(Temp_MA3, Insol_MA3) 포함
X_Vars = {'기온(°C)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)', '전운량(10분위)', ...
    'Temp_Avg', 'Temp_Min', 'Temp_Max', 'Is_Weekend', 'Is_Public_Holiday', ...
    'Dayoff', 'Monthly_Working_Days', 'SinHour', 'CosHour', 'Temp_MA3', 'Insol_MA3'};

X_Train_Mat = table2array(TblTrain(:, X_Vars));
X_Test_Mat  = table2array(TblTest(:, X_Vars));

[B_Lasso, FitInfo] = lasso(X_Train_Mat, Res_Train, 'CV', 10, 'Standardize', true);

idxLambdaMinMSE = FitInfo.IndexMinMSE;
optB = B_Lasso(:, idxLambdaMinMSE);

Valid_Idx = find(optB ~= 0);
Selected_Features = X_Vars(Valid_Idx);
fprintf('선택된 핵심 변수 개수: %d / %d\n', length(Selected_Features), length(X_Vars));
disp(Selected_Features');

X_Train_Sel = X_Train_Mat(:, Valid_Idx);
X_Test_Sel  = X_Test_Mat(:, Valid_Idx);

%% 5. Phase 3: LSBoost 앙상블 트리 모형 (잔차 예측)
fprintf('\n[Phase 3] LSBoost 모형으로 잔차의 비선형 패턴 학습 중...\n');

% 베이지안 최적화를 통한 하이퍼파라미터 튜닝
template_LS = templateTree('MinLeafSize', 5);
mdl_LSBoost = fitrensemble(X_Train_Sel, Res_Train, ...
    'Method', 'LSBoost', ...
    'OptimizeHyperparameters', {'LearnRate', 'NumLearningCycles'}, ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus', 'MaxObjectiveEvaluations', 15), ...
    'Learners', template_LS);

Res_Pred_Test = predict(mdl_LSBoost, X_Test_Sel);

%% 6. 최종 성능 평가 (2025년 미래 데이터 검증)
Final_Y_Pred = AR_Pred_Test + Res_Pred_Test;
True_Y = TblTest.Net_Load;

RMSE_AR = sqrt(mean((True_Y - AR_Pred_Test).^2));
MAPE_AR = mean(abs((True_Y - AR_Pred_Test) ./ True_Y)) * 100;

RMSE_Hybrid = sqrt(mean((True_Y - Final_Y_Pred).^2));
MAPE_Hybrid = mean(abs((True_Y - Final_Y_Pred) ./ True_Y)) * 100;

fprintf('\n=== 최종 모델 성능 평가 (2025년 Test Set) ===\n');
fprintf('[벤치마크: AR 단일 모형]   RMSE: %.2f, MAPE: %.2f%%\n', RMSE_AR, MAPE_AR);
fprintf('[제안모형: 하이브리드 융합] RMSE: %.2f, MAPE: %.2f%%\n', RMSE_Hybrid, MAPE_Hybrid);

%%
%% Phase 4: 유전 알고리즘(GA) 직접 구현을 통한 LSBoost 최적화 (Model B)
fprintf('\n=======================================================\n');
fprintf('[Phase 4] 유전 알고리즘(GA) 기반 하이퍼파라미터 탐색 시작\n');
fprintf('(※ 총 18회의 앙상블 학습이 진행되므로 10~15분 소요됩니다)\n');

% 1. GA 내부 검증셋 분할 (Data Leakage 방지)
% 2022~2023년을 훈련으로, 2024년을 검증(Validation)으로 사용하여 진화 적합도 평가
idx_Val = (year(TblTrain.Datetime) == 2024);
idx_Tr  = ~idx_Val;

X_GA_Tr  = X_Train_Sel(idx_Tr, :);
Y_GA_Tr  = Res_Train(idx_Tr);
X_GA_Val = X_Train_Sel(idx_Val, :);
Y_GA_Val = Res_Train(idx_Val);

% 2. GA 파라미터 초기 설정
popSize = 6;       % 한 세대의 개체 수
maxGen = 3;        % 최대 진화 세대 수
mutationRate = 0.2; % 돌연변이 확률 (20%)

% 초기 유전자 생성: [트리 개수(50~500), 학습률(0.01~0.5)]
pop = [randi([50, 500], popSize, 1), 0.01 + 0.49*rand(popSize, 1)];
bestParams = zeros(1, 2);
globalBestFitness = 0;

% 3. 진화 프로세스 (Evolution Loop)
for gen = 1:maxGen
    fprintf('▶ 세대 %d / %d 진행 중...\n', gen, maxGen);
    fitness = zeros(popSize, 1);

    % 개체별 적합도 평가 (Fitness Evaluation)
    for i = 1:popSize
        temp_mdl = fitrensemble(X_GA_Tr, Y_GA_Tr, 'Method', 'LSBoost', ...
            'NumLearningCycles', round(pop(i,1)), 'LearnRate', pop(i,2), ...
            'Learners', templateTree('MinLeafSize', 5));

        val_pred = predict(temp_mdl, X_GA_Val);
        rmse_val = sqrt(mean((Y_GA_Val - val_pred).^2));

        % 오차가 작을수록 적합도(Fitness)가 높도록 역수 취함
        fitness(i) = 1 / rmse_val; 

        if fitness(i) > globalBestFitness
            globalBestFitness = fitness(i);
            bestParams = pop(i, :);
        end
    end

    % 룰렛 휠 선택 (Roulette Wheel Selection)
    prob = fitness / sum(fitness);
    cumProb = cumsum(prob);
    newPop = zeros(popSize, 2);

    % 교차 및 돌연변이 (Crossover & Mutation)
    for i = 1:popSize
        parent1 = pop(find(cumProb >= rand(), 1), :);
        parent2 = pop(find(cumProb >= rand(), 1), :);

        % 교차: 두 부모의 유전자를 평균내어 자손 생성
        child = (parent1 + parent2) / 2;
        child(1) = round(child(1)); 

        % 돌연변이: 무작위 변이 발생
        if rand() < mutationRate
            child(1) = child(1) + randi([-30, 30]);      % 트리 개수 변이
            child(2) = child(2) + (rand()-0.5) * 0.1;    % 학습률 변이
        end

        % 논리적 경계 조건 처리
        child(1) = max(50, min(500, child(1)));
        child(2) = max(0.01, min(0.5, child(2)));

        newPop(i, :) = child;
    end
    pop = newPop;
end

fprintf('\n=> GA 최적 파라미터 도출 완료: 트리 개수 = %d, 학습률 = %.4f\n', bestParams(1), bestParams(2));

%% Phase 5: 성능 최종 대결 (Model A vs Model B)
% 도출된 최고 유전자로 최종 모델 훈련 (2022~2024년 전체 데이터 사용)
mdl_LSBoost_GA = fitrensemble(X_Train_Sel, Res_Train, 'Method', 'LSBoost', ...
    'NumLearningCycles', bestParams(1), 'LearnRate', bestParams(2), ...
    'Learners', templateTree('MinLeafSize', 5));

% 최종 테스트 셋(2025년) 예측 및 성과 산출
Res_Pred_Test_GA = predict(mdl_LSBoost_GA, X_Test_Sel);
Final_Y_Pred_GA = AR_Pred_Test + Res_Pred_Test_GA;

RMSE_GA = sqrt(mean((True_Y - Final_Y_Pred_GA).^2));
MAPE_GA = mean(abs((True_Y - Final_Y_Pred_GA) ./ True_Y)) * 100;

fprintf('\n========= [최종 결론] 최적화 알고리즘 성과 비교 =========\n');
fprintf('1. [Model A] 내장 함수(Bayesian) 최적화 : RMSE %.2f, MAPE %.2f%%\n', RMSE_Hybrid, MAPE_Hybrid);
fprintf('2. [Model B] 유전 알고리즘(GA) 직접 구현 : RMSE %.2f, MAPE %.2f%%\n', RMSE_GA, MAPE_GA);
fprintf('=========================================================\n');

%%%
%% Phase 6: 프로젝트 Quality 입증을 위한 시각화 검증 (Visualization)
fprintf('\n[Phase 6] 검증 시각화 그래프 생성 중...\n');

% 1. 실제값 vs 예측값 산점도 (Scatter Plot) 및 R-squared 시각화
figure('Name', 'Prediction Accuracy', 'Position', [100, 100, 600, 500]);
scatter(True_Y, Final_Y_Pred_GA, 10, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
% 45도 기준선 (완벽한 예측선)
maxVal = max(max(True_Y), max(Final_Y_Pred_GA));
minVal = min(min(True_Y), min(Final_Y_Pred_GA));
plot([minVal, maxVal], [minVal, maxVal], 'r--', 'LineWidth', 2);
xlabel('Actual Net Load (MW)');
ylabel('Predicted Net Load (MW)');
title('Actual vs Predicted (GA Hybrid Model)');
grid on; hold off;


% 2. 잔차(Residual) 분포 분석 (오차의 백색잡음 검증)
% 좋은 모델은 오차의 분포가 0을 중심으로 정규분포를 띄어야 함
Final_Residuals = True_Y - Final_Y_Pred_GA;
figure('Name', 'Residual Distribution', 'Position', [750, 100, 600, 500]);
histogram(Final_Residuals, 50, 'Normalization', 'pdf', 'FaceColor', [0.5 0.5 0.5]);
hold on;
% 정규분포 적합선 추가
pd = fitdist(Final_Residuals, 'Normal');
x_values = linspace(min(Final_Residuals), max(Final_Residuals), 100);
y_values = pdf(pd, x_values);
plot(x_values, y_values, 'r-', 'LineWidth', 2);
xlabel('Prediction Error (Residuals)');
ylabel('Probability Density');
title('Residual Error Distribution');
grid on; hold off;


% 3. 변수 중요도 (Feature Importance) - 교수님의 RandomForest.m 논리 차용
% GA로 학습된 최종 LSBoost 모델에서 어떤 변수가 예측에 기여했는지 추출
imp = predictorImportance(mdl_LSBoost_GA);
[impSorted, idx] = sort(imp, 'descend');

figure('Name', 'Feature Importance', 'Position', [400, 400, 700, 500]);
barh(impSorted(1:10)); % 상위 10개 변수만 출력
yticks(1:10);
yticklabels(Selected_Features(idx(1:10)));
set(gca, 'YDir', 'reverse'); % 1등이 맨 위에 오도록 뒤집기
xlabel('Relative Importance (Predictor Importance)');
title('Top 10 Feature Importance (LSBoost)');
grid on;


%% Phase 6: 프로젝트 Quality 입증을 위한 시각화 검증 (Visualization)
fprintf('\n[Phase 6] 검증 시각화 그래프 생성 중...\n');

% 1. 실제값 vs 예측값 산점도 (Scatter Plot) 및 R-squared 시각화
figure('Name', 'Prediction Accuracy', 'Position', [100, 100, 600, 500]);
scatter(True_Y, Final_Y_Pred_GA, 10, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
% 45도 기준선 (완벽한 예측선)
maxVal = max(max(True_Y), max(Final_Y_Pred_GA));
minVal = min(min(True_Y), min(Final_Y_Pred_GA));
plot([minVal, maxVal], [minVal, maxVal], 'r--', 'LineWidth', 2);
xlabel('Actual Net Load (MW)');
ylabel('Predicted Net Load (MW)');
title('Actual vs Predicted (GA Hybrid Model)');
grid on; hold off;


% 2. 잔차(Residual) 분포 분석 (오차의 백색잡음 검증)
% 좋은 모델은 오차의 분포가 0을 중심으로 정규분포를 띄어야 함
Final_Residuals = True_Y - Final_Y_Pred_GA;
figure('Name', 'Residual Distribution', 'Position', [750, 100, 600, 500]);
histogram(Final_Residuals, 50, 'Normalization', 'pdf', 'FaceColor', [0.5 0.5 0.5]);
hold on;
% 정규분포 적합선 추가
pd = fitdist(Final_Residuals, 'Normal');
x_values = linspace(min(Final_Residuals), max(Final_Residuals), 100);
y_values = pdf(pd, x_values);
plot(x_values, y_values, 'r-', 'LineWidth', 2);
xlabel('Prediction Error (Residuals)');
ylabel('Probability Density');
title('Residual Error Distribution');
grid on; hold off;


% 3. 변수 중요도 (Feature Importance) - 교수님의 RandomForest.m 논리 차용
% GA로 학습된 최종 LSBoost 모델에서 어떤 변수가 예측에 기여했는지 추출
imp = predictorImportance(mdl_LSBoost_GA);
[impSorted, idx] = sort(imp, 'descend');

figure('Name', 'Feature Importance', 'Position', [400, 400, 700, 500]);
barh(impSorted(1:10)); % 상위 10개 변수만 출력
yticks(1:10);
yticklabels(Selected_Features(idx(1:10)));
set(gca, 'YDir', 'reverse'); % 1등이 맨 위에 오도록 뒤집기
xlabel('Relative Importance (Predictor Importance)');
title('Top 10 Feature Importance (LSBoost)');
grid on;