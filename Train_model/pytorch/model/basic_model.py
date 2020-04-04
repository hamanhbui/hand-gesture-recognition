import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
def main():
    root='drive/My Drive/gesture_input/Dataset/3s_NML_magne/'
    train_x=np.load(root+'train_x.npy')
    train_y=np.load(root+'train_y.npy')
    valid_x=np.load(root+'valid_x.npy')
    valid_y=np.load(root+'valid_y.npy')
    test_x=np.load(root+'test_x.npy')
    test_y=np.load(root+'test_y.npy')

    # train_x=np.delete(train_x,6,axis=2)
    # train_x=np.delete(train_x,6,axis=2)
    # train_x=np.delete(train_x,6,axis=2)
    
    # valid_x=np.delete(valid_x,6,axis=2)
    # valid_x=np.delete(valid_x,6,axis=2)
    # valid_x=np.delete(valid_x,6,axis=2)
    
    # test_x=np.delete(test_x,6,axis=2)
    # test_x=np.delete(test_x,6,axis=2)
    # test_x=np.delete(test_x,6,axis=2)

    train_x=train_x.reshape(train_x.shape[0],-1)
    valid_x=valid_x.reshape(valid_x.shape[0],-1)
    test_x=test_x.reshape(test_x.shape[0],-1)
    train_y=np.where(train_y==1)[1]
    valid_y=np.where(valid_y==1)[1]
    test_y=np.where(test_y==1)[1]

    # pca = PCA()
    # pca.fit_transform(train_x)
    # k = 0
    # total = sum(pca.explained_variance_)
    # current_sum = 0
    # while(current_sum / total < 0.99):
    #     current_sum += pca.explained_variance_[k]
    #     k += 1
    # pca = PCA(n_components=k, whiten=True)
    # train_x = pca.fit_transform(train_x)
    # valid_x = pca.transform(valid_x)
    # test_x=pca.transform(test_x)

    model=GaussianNB()
    # model=tree.DecisionTreeClassifier()
    # model = RandomForestClassifier()
    # model = KNeighborsClassifier()
    # model = SVC()
    model.fit(train_x, train_y)

    y_pred = model.predict(train_x)
    accuracy = accuracy_score(train_y, y_pred)
    print(accuracy)

    y_pred = model.predict(valid_x)
    accuracy = accuracy_score(valid_y, y_pred)
    print(accuracy)

    y_pred = model.predict(test_x)
    accuracy = accuracy_score(test_y, y_pred)
    print(accuracy)

    # print(confusion_matrix(test_y,y_pred))
    # print(classification_report(test_y,y_pred))

main()