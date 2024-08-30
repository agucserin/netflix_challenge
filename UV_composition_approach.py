import numpy as np
import sys

def load_movie_data(path):
    # 파일에서 데이터 로드
    with open(path, 'r', encoding='utf-8') as f:
        i = 0
        lines = f.readlines()
        for line in lines:
            movie = int(line.strip().split(',')[0])
            dicMovie[movie] = i
            dicIdxtoMovie[i] = movie
            i += 1
    return i

def load_user_data(path):
    dic_user = {}
    i = 0
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lis =  line.strip().split(',')
            user = int(lis[0]) - 1
            if user not in dic_user:
                dic_user[user] = 1
                i += 1
    return i

def load_train_data(path):
    # 파일에서 데이터 로드
    train_list = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip().split(',')
            train_list.append([int(items[0]) - 2, int(items[1]), float(items[2])])
    return train_list

def split_train_validation(train_list, validation_size, split_idx):
    # 검증 세트와 훈련 세트 나누기
    split_idx = validation_size * split_idx
    validation_set = train_list[split_idx:split_idx + validation_size]
    train_set = train_list[:split_idx] + train_list[split_idx + validation_size:]
    for data in train_set:
        user, movie, rating = data
        A[user][dicMovie[movie]] = rating
    return validation_set

def load_test_data(path):
    # 파일에서 데이터 로드
    lis = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip().split(',')
            lis.append([int(items[0]) - 2, int(items[1])])
    return lis
    
def get_truth_matrix(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lis =  line.strip().split(',')
            user, movie, rating = int(lis[0]) - 1, int(lis[1]), float(lis[2])
            A_truth[user][dicMovie[movie]] = rating

def train(learning_rate, num_iterations, lambda_reg, validation_set, patience=10):
    # 초기화
    mu = np.mean(A[A > 0])  # 전체 평균 평점
    bu = np.zeros(num_users)  # 사용자 바이어스
    bi = np.zeros(num_movies)  # 영화 바이어스
    best_a = float('inf')
    best_iteration = 0
    patience_counter = 0
    # Gradient Descent를 통한 최적화
    for iteration in range(num_iterations):
        for i in range(num_users):
            for j in range(num_movies):
                if A[i, j] > 0:  # 기존 평점이 있는 경우에만 업데이트
                    error_ij = 2*(A[i, j] - (mu + bu[i] + bi[j] + np.dot(U[i, :], V[:, j])))
                    bu[i] += learning_rate * (error_ij - lambda_reg * bu[i])
                    bi[j] += learning_rate * (error_ij - lambda_reg * bi[j])
                    U[i, :] += learning_rate * (error_ij * V[:, j] - lambda_reg * U[i, :])
                    V[:, j] += learning_rate * (error_ij * U[i, :] - lambda_reg * V[:, j])

        A_pred = mu + bu[:, np.newaxis] + bi[np.newaxis, :] + np.dot(U, V)
        rmse = 0
        numb = 0
        for i in range(num_users):
            for j in range(num_movies):
                if A[i, j] > 0:
                    rmse += (A[i, j] - A_pred[i, j]) ** 2
                    numb += 1
        rmse = (rmse / numb) ** (1/2)
        a = validation_test(A_pred,validation_set)
        if iteration % 20 == 0:
            print("Test result_" +str(iteration)+ ": "+ str(a))
            f.write(f"Test result_{iteration}: {a}\n")
        if a < best_a:
            best_a = a
            best_iteration = iteration
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"k {K} lambda {lambda_reg}: {best_iteration}. Best RMSE: {best_a}")
            f.write(f"k {K} lambda {lambda_reg}: {best_iteration}. Best RMSE: {best_a}\n")
            break
    return A_pred, best_a

def test(A_pred):
    numb = len(test_list)
    rmse = 0
    for data in test_list:
        i, j = data[0] - 2, dicMovie[data[1]]
        rmse += (A_truth[i, j] - A_pred[i, j]) ** 2
    return (rmse / numb) ** (1/2)

def validation_test(A_pred, valdiation_list):
    numb = len(valdiation_list)
    rmse = 0
    for data in valdiation_list:
        i, j = data[0], dicMovie[data[1]]
        rmse += (A_truth[i, j] - A_pred[i, j]) ** 2
    return (rmse / numb) ** (1/2)

if __name__ == "__main__":
    movie_path = sys.argv[1]
    train_path = sys.argv[2]
    test_path = sys.argv[3]
    ground_truth_path = sys.argv[4]

    dicMovie = {}
    dicIdxtoMovie = {}
    num_movies = load_movie_data(movie_path)
    num_users = load_user_data(ground_truth_path)
    test_list = load_test_data(test_path)

    A_truth = np.zeros((num_users, num_movies))

    get_truth_matrix(ground_truth_path)
    train_list = load_train_data(train_path)

    np.random.seed(42)
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
                f.write(f"@@@ {K} {lambda_reg} {total_result/6}")

#python netflix_0703.py movies.txt ratings.txt ratings_test.txt ground_truth.txt
