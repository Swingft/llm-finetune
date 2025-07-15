from graphcodebert_utils import (
    build_dependency_graph_from_json,
    get_graphcodebert_embeddings,
    train_regression_model,
    predict_and_rank
)

# 1. 의존성 그래프 생성
labels = build_dependency_graph_from_json("output.json")

# 2. 임베딩 추출
dataset = get_graphcodebert_embeddings(labels)

# 3. 회귀 모델 학습
model, X = train_regression_model(dataset, epochs=100)

# 4. 예측 및 출력
predict_and_rank(model, X, labels)
