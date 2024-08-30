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

def load_train_data(path):
    # 파일에서 데이터 로드
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            user, movie, rating, timestamp = line.strip().split(',')
            user, movie, rating = int(user), int(movie), float(rating)
            A[user-1][dicMovie[movie]] = rating

def load_test_data(path):
    # 파일에서 데이터 로드
    lis = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip().split(',')
            lis.append([int(items[0]), int(items[1])])
    return lis
    
def get_truth_matrix(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lis =  line.strip().split(',')
            user, movie, rating = int(lis[0]), int(lis[1]), float(lis[2])
            A_truth[user][dicMovie[movie]] = rating

def train(learning_rate, num_iterations, lambda_reg):
    # 초기화
    mu = np.mean(A[A > 0])  # 전체 평균 평점
    bu = np.zeros(num_users)  # 사용자 바이어스
    bi = np.zeros(num_items)  # 영화 바이어스
    best_a = float('inf')
    best_iteration = 0
    patience_counter = 0
    # Gradient Descent를 통한 최적화
    filename = "result_0820_" + str(K) + "_" + str(lambda_reg) + ".txt"
    with open(filename, "w") as f:
        for iteration in range(num_iterations):
            print(iteration, end=" : ")
            for i in range(num_users):
                for j in range(num_items):
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
                for j in range(num_items):
                    if A[i, j] > 0:
                        rmse += (A[i, j] - A_pred[i, j]) ** 2
                        numb += 1
            rmse = (rmse / numb) ** (1/2)
            print("RMSE : ", rmse)
            f.write(f"Iteration: {iteration}; RMSE: {rmse}\n")
        a = test(A_pred)
        print(a)
        f.write(f"Test result: {a}\n")
        return A_pred

def test(A_pred):
    numb = len(test_list)
    rmse = 0
    for data in test_list:
        i, j = data[0] - 1, dicMovie[data[1]]
        rmse += (A_truth[i, j] - A_pred[i, j]) ** 2
    return (rmse / numb) ** (1/2)

if __name__ == "__main__":
    import sys

    movie_path = sys.argv[1]
    train_path = sys.argv[2]
    test_path = sys.argv[3]
    ground_truth_path = sys.argv[4]

    dicMovie = {}
    dicIdxtoMovie = {}
    movie_numb = load_movie_data(movie_path)
    test_list = load_test_data(test_path)

    A = np.zeros((1000, movie_numb))
    A_truth = np.zeros((1000, movie_numb))

    get_truth_matrix(ground_truth_path)
    load_train_data(train_path)

    num_users, num_items = A.shape
    K = 7  # 잠재 요인 수

    np.random.seed(42)
    U = np.random.rand(num_users, K)
    V = np.random.rand(K, num_items)

    learning_rate = 0.0005
    num_iterations = 1600
    lambda_reg = 0.34

    A_pred = train(learning_rate, num_iterations, lambda_reg)

#python UV_composition_test.py movies.txt ratings.txt ratings_test.txt ground_truth.txt
