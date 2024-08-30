# 1. UV 분해를 활용한 넷플릭스 영화 추천 시스템 구현

## 1. 개요

### **1. 프로젝트 배경**

 이 프로젝트는 2023년 가을학기에 수강한 ‘빅데이터 분석 개론(EE412)’ 과목에서 수행한 과제 중 하나로, 넷플릭스가 주최했던 넷플릭스 챌린지(Netflix Prize)에서 영감을 받았습니다. 넷플릭스 챌린지는 2006년에 시작된 대회로, 전 세계의 데이터 과학자들이 사용자에게 적합한 영화를 추천하는 알고리즘을 개발하여 가장 높은 예측 정확도를 달성하기 위해 경쟁했습니다. 이 대회는 추천 시스템 분야에서 혁신적인 발전을 이루었으며, 오늘날에도 널리 사용되고 있는 다양한 기법들이 여기서 개발되었습니다.

### **2. 프로젝트 목적**

 이 프로젝트의 주요 목적은 넷플릭스 챌린지에서 사용된 데이터셋을 기반으로 영화 추천 시스템을 구현하는 것입니다. 이를 통해 사용자에게 맞춤형 영화를 추천하고, 모델의 예측 정확도를 높여 RMSE(Root Mean Squared Error)를 최소화하는 것을 목표로 삼았습니다. 실제 산업 환경에서의 활용 가능성을 고려하여, 다양한 모델링 기법과 최적화 전략을 적용하였습니다.

### **3. 접근 방식**

- **나이브 접근법 vs UV 분해 접근법** : 크게 2가지 방식을 사용해 예측 시스템을 구현했습니다.
- **모델링**: 기본적인 행렬 분해 기법을 사용하여 사용자와 영화 간의 상호작용을 모델링했습니다. 또한, 사용자와 영화의 바이어스를 고려한 고급 모델을 구현하여, 더 정확한 추천 결과를 제공할 수 있도록 최적화했습니다. 다양한 하이퍼파라미터 튜닝을 통해 모델의 성능을 극대화하였습니다.
- **검증 및 최적화**: 검증 세트를 활용하여 모델의 성능을 평가하고, 그리드 검색(Grid Search)과 같은 방법을 사용하여 최적의 하이퍼파라미터를 찾았습니다. 최적화된 모델을 통해 예측 성능을 지속적으로 개선했습니다.
- **테스트 및 평가**: 최종적으로, 학습된 모델을 테스트 데이터에 적용하여 예측 정확도를 평가했으며, RMSE를 기준으로 모델 성능을 검증했습니다.

### **4. 사용 기술 및 도구**

- **프로그래밍 언어**: Python
- **데이터 분석 도구**: Pandas, NumPy
- **모델링 및 최적화**: Scikit-learn, Matrix Factorization, Stochastic Gradient Descent (SGD)
- **시각화 도구**: Matplotlib, Seaborn

### **5. 기대 효과 및 비즈니스 임팩트**

 이 프로젝트는 넷플릭스와 같은 스트리밍 플랫폼에서 개인화된 사용자 경험을 제공하는 데 중요한 기여를 할 수 있습니다. 정확한 영화 추천 시스템은 사용자의 만족도를 높이고, 플랫폼에서의 체류 시간을 증가시킬 수 있습니다. 또한, 추천 시스템의 성능을 향상시키는 것은 사용자가 자주 이용하지 않던 콘텐츠를 발견하게 하여, 콘텐츠 소비 패턴을 다양화하고, 비즈니스 성장을 촉진할 수 있습니다.

## 2. 실험 내용

### 1. 나이브 접근법: 사용자 및 영화 평균 기반 추천 시스템

1. **개요**
    
    Naive approach에서는 사용자와 영화의 개별적인 특성을 반영하지 않고, 단순히 사용자와 영화의 평균 평점을 기반으로 예측을 수행합니다. 이 방법은 기본적인 영화 추천 시스템을 구성할 때 시작점으로 사용될 수 있으며, 복잡한 모델을 도입하기 전에 기본적인 성능을 평가하기 위해 유용합니다.
    
2. **접근방식**
    
    이 naive approach는 다음과 같은 단계로 진행됩니다:
    
    - **데이터 수집**: 주어진 데이터 파일에서 사용자, 영화, 평점 데이터를 수집합니다.
    - **사용자 및 영화 평균 계산**: 각 사용자의 평균 평점과 각 영화의 평균 평점을 계산합니다.
    - **평점 예측**: 특정 사용자-영화 쌍에 대해 사용자의 평균 평점과 영화의 평균 평점을 결합하여 예측 평점을 계산합니다.
    - **RMSE 평가**: 예측된 평점과 실제 평점 간의 차이를 통해 RMSE를 계산하여 모델의 성능을 평가합니다.
    
3. **핵심 코드 설명**
    
    **<사용자 및 영화 평균 계산>**
    
    ```python
    user_avg = [0 for x in range(i0)]
    movie_avg = [0 for x in range(i1)]
    
    # 사용자 평균 계산
    for i in range(i0):
        user = real_matrix[i]
        tot_rate = 0
        tot_gaso = 0
        for j in range(i1):
            if user[j][0] != 0:
                tot_gaso += 1
                tot_rate += user[j][0]
        avg = tot_rate/tot_gaso
        user_avg[i] = avg
    
    # 영화 평균 계산
    for i in range(i1):
        tot_rate = 0
        tot_gaso = 0
        for j in range(i0):
            if real_matrix[j][i][0] != 0:
                tot_rate += real_matrix[j][i][0]
                tot_gaso += 1
        avg = tot_rate/tot_gaso
        movie_avg[i] = avg
    ```
    
    
    - 각 사용자의 평균 평점을 계산하여 `user_avg` 리스트에 저장합니다.
    - 각 영화의 평균 평점을 계산하여 `movie_avg` 리스트에 저장합니다.
    
    **<평점 예측 및 평가>**
    
    ```python
    def check(true_matrix):
        rmse = 0
        with open(sys.argv[2], 'r') as file:
            for line in file:
                items = line.strip().split(',')
                row = [int(items[0]), int(items[1]) , int(items[3])]
                if row[1] not in dicMovie:
                    a = user_avg[dicUser[row[0]]]
                    rmse += (a - true_matrix[row[0] - 2][row[1] - 1]) ** 2
                else:
                    a = user_avg[dicUser[row[0]]]
                    b = movie_avg[dicMovie[row[1]]]
                    rmse += ((a + b)/2 - true_matrix[row[0] - 2][row[1] - 1]) ** 2
                
        print(math.sqrt(rmse / 10000))
    ```
    
    - `check` 함수는 각 사용자-영화 쌍에 대해 예측된 평점과 실제 평점 간의 RMSE를 계산합니다.
    - 특정 영화가 영화 목록에 없을 경우, 사용자 평균만을 사용하여 예측을 수행합니다.
    - RMSE는 예측 성능을 평가하는 중요한 지표로, 값이 낮을수록 예측이 정확함을 의미합니다.
    
4. **결과 및 한계**
    
    이 Naive Approach는 단순한 방법으로, 각 사용자와 영화의 평균 평점을 결합하여 평점을 예측합니다. 이 접근 방식은 매우 간단하고 빠르게 계산할 수 있지만, 다음과 같은 한계점이 있습니다:
    
    - **개별적인 특성 반영 부족**: 사용자의 선호도나 영화의 장르, 시간에 따른 변화 등을 고려하지 않습니다.
    - **과도한 일반화**: 모든 사용자가 비슷한 선호도를 가진다고 가정하는 과도한 일반화가 발생할 수 있습니다.
    - **성능 한계**: 테스트 결과 RMSE가 **0.9183**로 나타났으며, 이는 복잡한 모델에 비해 성능이 낮습니다.
    
    이 Naive Approach의 한계를 극복하고, 보다 정교한 예측을 위해 **UV Composition**과 같은 복잡한 모델을 다음 단계에서 도입하였습니다.
    

### 2. UV 분해 접근법

1. **개요**
    
     UV Composition은 추천 시스템에서 널리 사용되는 기법 중 하나로, 사용자와 영화 간의 상호작용을 잠재 요인(latent factors)으로 분해하여 예측 모델을 구축하는 방식입니다. 이 방식은 행렬 분해(Matrix Factorization) 기법을 사용하여 사용자와 영화 간의 관계를 저차원 공간에 표현하고, 이를 통해 사용자에게 적합한 영화를 추천합니다. 이 프로젝트에서는 넷플릭스 데이터셋을 기반으로 UV Composition 기법을 사용하여 영화 추천 시스템을 구현했습니다.
    
2. **접근방식**
    
    이 UV Composition 방식은 다음과 같은 단계로 진행됩니다:
    
    - **데이터 수집 및 전처리**: 주어진 데이터 파일에서 사용자, 영화, 평점 데이터를 수집하고, 이 데이터를 기반으로 사용자와 영화의 잠재 요인 행렬 U와 V를 초기화합니다.
    - **모델 학습**: 사용자와 영화의 바이어스 및 잠재 요인 행렬을 학습하기 위해 확률적 경사 하강법(Stochastic Gradient Descent, SGD)을 사용하여 최적화합니다.
    - **모델 검증 및 최적화**: 학습된 모델을 검증 세트(validation set)에서 평가하고, 성능이 가장 좋은 모델을 선택하기 위해 조기 종료(Early Stopping) 기법을 사용합니다.
    - **테스트 및 평가**: 최적의 하이퍼파라미터로 학습된 모델을 테스트 세트(test set)에 적용하여 최종 성능을 평가합니다.
    
3. **핵심 코드 설명**
    
    **<데이터 로드 및 초기화>**
    
    ```python
    def load_movie_data(path):
        with open(path, 'r', encoding='utf-8') as f:
            i = 0
            lines = f.readlines()
            for line in lines:
                movie = int(line.strip().split(',')[0])
                dicMovie[movie] = i
                dicIdxtoMovie[i] = movie
                i += 1
        return i
    
    def load_train_data(path):
        train_list = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.strip().split(',')
                train_list.append([int(items[0]) - 2, int(items[1]), float(items[2])])
        return train_list
    ```
    
    
    - `load_movie_data` 함수는 영화 데이터를 로드하여 `dicMovie` 사전에 영화 ID와 인덱스를 매핑합니다.
    - `load_train_data` 함수는 훈련 데이터를 로드하여 각 사용자의 평점 정보를 리스트에 저장합니다.
    
    **<훈련 및 최적화>**
    
    ```python
    def train(learning_rate, num_iterations, lambda_reg, validation_set, patience=10):
        mu = np.mean(A[A > 0])  # 전체 평균 평점
        bu = np.zeros(num_users)  # 사용자 바이어스
        bi = np.zeros(num_movies)  # 영화 바이어스
        best_a = float('inf')
        patience_counter = 0
        
        for iteration in range(num_iterations):
            for i in range(num_users):
                for j in range(num_movies):
                    if A[i, j] > 0:
                        error_ij = 2*(A[i, j] - (mu + bu[i] + bi[j] + np.dot(U[i, :], V[:, j])))
                        bu[i] += learning_rate * (error_ij - lambda_reg * bu[i])
                        bi[j] += learning_rate * (error_ij - lambda_reg * bi[j])
                        U[i, :] += learning_rate * (error_ij * V[:, j] - lambda_reg * U[i, :])
                        V[:, j] += learning_rate * (error_ij * U[i, :] - lambda_reg * V[:, j])
    
            A_pred = mu + bu[:, np.newaxis] + bi[np.newaxis, :] + np.dot(U, V)
            a = validation_test(A_pred, validation_set)
            
            if iteration % 20 == 0:
                print(f"Test result_{iteration}: {a}")
    
            if a < best_a:
                best_a = a
                patience_counter = 0
            else:
                patience_counter += 1
    
            if patience_counter >= patience:
                print(f"Early stopping at iteration {iteration}; Best RMSE: {best_a}")
                break
    
        return A_pred, best_a
    ```
    
    - `train` 함수는 사용자와 영화의 바이어스 및 잠재 요인 행렬을 SGD를 통해 최적화합니다.
    - 모델의 학습 과정 중 일정 간격으로 검증 세트에서의 성능을 측정하며, 성능이 향상되지 않으면 조기 종료합니다.
    
    **<평가 및 검증>**
    
    ```python
    def validation_test(A_pred, validation_list):
        numb = len(validation_list)
        rmse = 0
        for data in validation_list:
            i, j = data[0], dicMovie[data[1]]
            rmse += (A_truth[i, j] - A_pred[i, j]) ** 2
        return (rmse / numb) ** (1/2)
    ```
    
    - `validation_test` 함수는 검증 세트에서 모델의 성능을 평가하여 RMSE를 계산합니다.
    - 이 함수는 모델의 예측이 얼마나 정확한지를 평가하는 데 사용되며, 최적의 하이퍼파라미터를 찾는 데 중요한 역할을 합니다.
    
    **<전체 모델 평가 및 최적 하이퍼파라미터 탐색>**
    
    ```python
    if __name__ == "__main__":
        filename = f"result_0703.txt"
        with open(filename, "a") as f:
            for i0 in range(0,3):
                for j0 in range(5,8):
                    total_result = 0
                    for k0 in range(0,6):
                        A = np.zeros((num_users, num_movies))
    
                        K = j0
                        U = np.random.rand(num_users, K)
                        V = np.random.rand(K, num_movies)
    
                        valid_set = split_train_validation(train_list, 15000, k0)
                        learning_rate = 0.0005
                        num_iterations = 40000
                        lambda_reg = 0.09 + (i0 * 0.05)
    
                        A_pred, result = train(learning_rate, num_iterations, lambda_reg, valid_set)
                        total_result += result
                    print(f"@@@ {K} {lambda_reg} {total_result/6}")
                    f.write(f"@@@ {K} {lambda_reg} {total_result/6}\n")
    ```
    
    - 다양한 하이퍼파라미터 조합(K 값, 학습률, 정규화 파라미터)을 시도하여 최적의 성능을 찾습니다.
    - 각 조합에 대해 평균 RMSE를 계산하여 최적의 하이퍼파라미터를 선택합니다.
    
4. **결과 및 한계**
    
     아래 표는 각기 다른 하이퍼파라미터 조합(K와 λ 값)으로 학습된 모델이 여러 검증 세트(validation set)에서 평가된 결과(RMSE)의 평균을 나타냅니다. 여기서 **K**는 잠재 요인의 개수를, **λ** (lambda)는 정규화 파라미터를 의미합니다.
    
    | K \ λ | 0.09 | 0.14 | 0.19 | 0.24 | 0.29 | 0.34 |
    | --- | --- | --- | --- | --- | --- | --- |
    | **5** | 0.8965 | 0.8919 | 0.8874 | 0.8855 | 0.8834 | 0.8838 |
    | **6** | 0.8993 | 0.8941 | 0.8883 | 0.8859 | 0.8842 | 0.8833 |
    | **7** | 0.9005 | 0.8938 | 0.8889 | 0.8854 | 0.8841 | **0.8831** |
    
    | K \ λ | 0.39 | 0.44 | 0.49 | 0.54 | 0.59 | 0.64 |
    | --- | --- | --- | --- | --- | --- | --- |
    | **5** | 0.8844 | 0.8855 | 0.886 | 0.8871 | 0.8881 | 0.8892 |
    | **6** | 0.8843 | 0.885 | 0.8858 | 0.887 | 0.8876 | 0.8889 |
    | **7** | 0.8846 | 0.8848 | 0.8855 | 0.8865 | 0.8875 | 0.8885 |
    
     표에서 볼 수 있듯이, 실험 결과 **K** = 7, **λ** = 0.34일 때, 평균 RMSE가 **0.8831**로 가장 낮게 나타났습니다. 이 조합은 다양한 하이퍼파라미터 조합 중에서 가장 우수한 성능을 보였습니다. 따라서 이 최적의 하이퍼파라미터를 사용하여 테스트 세트에서 모델의 성능을 평가하였습니다.
    
     테스트 세트에서 측정한 결과, 최종 RMSE는 **0.8734**로 나타났습니다. 이는 검증 세트에서의 성능과 비교하여 일관성 있는 결과를 보여주며, 모델이 잘 일반화되었음을 나타냅니다.
    
     이 결과를 통해, UV Composition 방식이 추천 시스템의 성능을 최적화하는 데 효과적임을 확인할 수 있었습니다. 그러나, 이 방법은 하이퍼파라미터 선택에 민감하며, 최적의 성능을 달성하기 위해서는 충분한 검증 과정이 필요합니다. 또한, 복잡한 모델일수록 학습 시간이 길어질 수 있으며, 이에 따른 계산 비용 역시 중요한 고려사항입니다.
    

## 3. 결론

 UV Composition 방식은 Naive Approach에 비해 훨씬 더 높은 성능을 제공하며, 그 결과 RMSE가 상당히 개선되었습니다. Naive Approach의 RMSE는 0.9183이었지만, UV Composition을 적용한 모델에서는 0.8734로 감소하여, 약 **4.89%**의 성능 향상을 이뤄냈습니다. 이 차이는 사용자와 영화 간의 상호작용을 더 정교하게 모델링할 수 있는 UV Composition의 강점을 잘 보여줍니다.

 비록 UV Composition 방식은 Naive Approach에 비해 더 높은 계산 복잡성과 학습 시간을 요구하지만, 그만큼 더 정확한 추천 결과를 제공할 수 있습니다. 이 프로젝트를 통해 얻은 경험은 추천 시스템의 성능을 최적화하기 위한 중요한 인사이트를 제공하며, 실제 비즈니스 응용에서 고객 만족도를 높이고, 이탈률을 줄이는 데 유용하게 사용될 수 있습니다.

 나아가, 이 프로젝트는 향후 더 복잡한 모델, 예를 들어 딥러닝 기반의 추천 시스템이나 하이브리드 추천 시스템으로 확장할 수 있는 기반을 마련해줍니다. 이로써 데이터 기반 의사결정과 맞춤형 사용자 경험 제공에 기여할 수 있는 더욱 발전된 기술적 토대를 마련할 수 있을 것입니다.
