import csv
import pickle
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


##load data
load_data = open('data_film.csv', 'r')
data_reader = csv.reader(load_data,delimiter='')
data= list(data_reader)


# selecte train data and test data
test = data[:20000]
train = data[20000:]


user = []
item = []

for i in range(0,len(train)):
    user.append(int(train[i][0]))
    item.append(int(train[i][1]))



## create matrix for rating
rating_matrix = np.zeros((max(user)+1,max(item)+1))
for i in range(0,len(train)):
    rating_matrix[user[i]][item[i]]=int(train[i][2])


##ceck the common list
def common(mylist):
    for i in range(0,len(mylist)-1):
        if mylist[i]!=mylist[i+1]:
            return False
    return True

#check the corelaton between two user
def corelation_two_user(user1, user2):
    user1_items = []
    user2_items = []
    for i in range(1, len(rating_matrix[user1])):
        if (rating_matrix[user1][i] != 0 and rating_matrix[user2][i] != 0):
            user1_items.append(rating_matrix[user1][i])
            user2_items.append(rating_matrix[user2][i])
    if len(user2_items)==0 or len(user2_items)==1 or len(user2_items)==2 : return 0
    if common(user2_items)==True or common(user1_items)==True: return 0
    return pearsonr(user1_items, user2_items)[0]



##create matrix of corelation
corelation_two_user_matrix = np.zeros((len(rating_matrix), len(rating_matrix)))
for i in range(1, len(corelation_two_user_matrix)):
    for j in range(i, len(corelation_two_user_matrix)):
        corelation_two_user_matrix[i][j] = corelation_two_user(i, j)
        corelation_two_user_matrix[j][i] = corelation_two_user_matrix[i][j]
print("Coralation between two user:\n------------------------------------------\n",corelation_two_user_matrix)
print("")


#calculate the average for rating matrix
average=[]
for i in range(0,len(rating_matrix)):
    sum_rate=0
    c=0
    for j in range(0,len(rating_matrix[0])):
        if rating_matrix[i][j]!=0:
            c=c+1
            s_rate = sum_rate + rating_matrix[i][j]
    if c==0: average.append(0)
    else: average.append(s_rate/c)

###according to 100 neighbor we calculate rating matrix for user(100 closet neighbor)
neighbor_size = 100
def neighbor(user,item):
    list_nbr = []
    for i in range(1, len(corelation_two_user_matrix)):
        if rating_matrix[i][item] != 0 and i != user :
            list_nbr.append([corelation_two_user_matrix[user][i],rating_matrix[i][item]])
    list_nbr = sorted(list_nbr, reverse=True)
    return list_nbr

#predict rate of user according to neighbor
def prediction(user,item):
    s = neighbor(user, item)[:100]
    if s ==[]:
        return average[user]
    s1 = []
    sum1=0
    for i in range(0,len(s)):
            s1.append(s[i][0])
            sum1=sum1+s[i][0]*s[i][1]##calculate weight avrege
    if sum(s1)==0: return average[user]
    return sum1/sum(s1)


predicated_rating_matrix=np.zeros((len(rating_matrix),len(rating_matrix[0])))
for user in range(1,len(rating_matrix)):
    for item in range(1, len(rating_matrix[0])):
         if rating_matrix[user][item]==0.0: predicated_rating_matrix[user][item]=prediction(user,item)
         else: predicated_rating_matrix[user][item]=rating_matrix[user][item]

print("predicated rating matrix :\n-----------------------------------------\n",predicated_rating_matrix)


#create SVD
def delete_column(data,k):
    d=[]
    for i in range(0,len(data)):
        d.append(data[i][:k])
    return d

def delete_row(data,k):
    d=data[:k]
    return d


#decomposation rating matrix
acc_list = []
k_list = []
for k in range(10,200,1):

    X = np.array(predicated_rating_matrix)
    P, D, Q = np.linalg.svd(X, full_matrices=False)
    D = np.diag(D)
    P=delete_column(P,k)
    D=delete_column(D,k)
    D=delete_row(D,k)
    Q=delete_row(Q,k)

    P=np.array(P)
    D=np.array(D)
    Q=np.array(Q)

    predicted_rating_matrix_svd = np.dot(np.dot(P,D),Q)

    aquracy=[]
    for test_intracton in test:
        i=int(test_intracton[0])
        j=int(test_intracton[1])
        aquracy.append(abs(predicted_rating_matrix_svd[i][j]-int(test_intracton[2])))
    acc_list.append((sum(aquracy)/len(aquracy)))
    k_list.append(k)

print("accuracy list for vareity of k :\n------------------------------------\n",acc_list)
plt.plot(k_list, acc_list , "r")
plt.xlabel('demation of k')
plt.ylabel('Accuracy')
plt.show()