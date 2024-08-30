import sys
import math
import numpy as np

def get_matrix(file_name):
    global user_avg
    global movie_avg
    global real_matrix
    with open(file_name, 'r') as file:
        i0 = 0
        i1 = 0
        for line in file:
            lis = parse_line(line)
            if lis[0] not in dicUser:
                dicUser[lis[0]] = i0
                dicIdxtoUser[i0] = lis[0]
                i0 += 1
            if lis[1] not in dicMovie:
                dicMovie[lis[1]] = i1
                dicIdxtoMovie[i1] = lis[1]
                i1 += 1

    real_matrix = [[[0 for _ in range(2)] for _ in range(i1)] for _ in range(i0)]

    user_avg = [0 for x in range(i0)]
    movie_avg = [0 for x in range(i1)]

    with open(file_name, 'r') as file:
        for line in file:
            lis = parse_line(line)
            real_matrix[dicUser[lis[0]]][dicMovie[lis[1]]][0] = lis[2]
            real_matrix[dicUser[lis[0]]][dicMovie[lis[1]]][1] = lis[3]

    util_matrix = [[0 for _ in range(i1)] for _ in range(i0)]

    #prtMat(real_matrix)

    for i in range(i0):
        user = real_matrix[i]
        tot_rate = 0
        tot_gaso = 0
        for j in range(i1):
            if user[j][0] != 0:
                tot_gaso += 1
                tot_rate += user[j][0]
        avg = tot_rate/tot_gaso
        for j in range(i1):
            if user[j][0] != 0:
                util_matrix[i][j] = user[j][0]
            else:
                util_matrix[i][j] = avg
        user_avg[i] = avg

    for i in range(i1):
        tot_rate = 0
        tot_gaso = 0
        for j in range(i0):
            if real_matrix[j][i][0] != 0:
                tot_rate += real_matrix[j][i][0]
                tot_gaso += 1
        avg = tot_rate/tot_gaso
        movie_avg[i] = avg

    return util_matrix

def get_truth_matrix(file_name):
    data=[]

    with open(file_name, 'r') as file:
        for line in file:
            items = line.strip().split(',')
            row = [int(items[0]), int(items[1]), float(items[2]), int(items[3])]
            data.append(row)
        data=np.array(data)
    usernumber=max(set(data[:,0]))
    movienumber=max(set(data[:,1]))
    util_matrix=np.zeros((int(usernumber),int(movienumber)))
    for profile in data:
        user,item,rate,time=profile
        util_matrix[int(user)-1][int(item)-1]=rate
    
    return util_matrix

def parse_line(line):
    lines = []
    tmp = line.split(",")
    for x in range(4):
        if x < 2:
            lines.append(int(tmp[x]))
        else:
            lines.append(float(tmp[x]))
    return lines

def check(true_matrix):
    rmse = 0
    with open(sys.argv[2], 'r') as file:
        for line in file:
            items = line.strip().split(',')
            row = [int(items[0]), int(items[1]) , int(items[3])]
            b = 0
            if row[1] not in dicMovie:
                a = user_avg[dicUser[row[0]]]
                rmse += (a - true_matrix[row[0] - 2][row[1] - 1]) ** 2
                #print(a,true_matrix[row[0] - 2][row[1] - 1])
            else:
                a = user_avg[dicUser[row[0]]]
                b = movie_avg[dicMovie[row[1]]]
                rmse += ((a + b)/2 - true_matrix[row[0] - 2][row[1] - 1]) ** 2
                #print((a + b) / 2,true_matrix[row[0] - 2][row[1] - 1])
            
    print(math.sqrt(rmse / 10000))

dicUser = {}
dicIdxtoUser = {}
dicMovie = {}
dicIdxtoMovie = {}
if __name__ == "__main__":

    umatrix = get_matrix(sys.argv[1])
    true_matrix = get_truth_matrix(sys.argv[3])

    check(true_matrix)