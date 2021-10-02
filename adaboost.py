from numpy import *

'''
    生成数据
'''
def createData():
    dataMat = matrix(
        [[0.],[1.],[2.],[3.],[4.],[5.],[6.],[7.],[8.],[9.]])
    classLabels = [1.0,1.0,1.0,-1.0,-1.0,-1.0,1.0,1.0,1.0,-1.0]
    return dataMat, classLabels

'''
    根据当前阈值，得出估计结果
'''
def tryThreshold(dataMat,dimen,threshold,direction):
    m,n = shape(dataMat)
    predictedRes = ones((m,1))
    if direction == 'front':
        for i in range(m):
            if dataMat[:,dimen][i]<=threshold:
                predictedRes[i] = -1.0
    else:
        for i in range(m):
            if dataMat[:,dimen][i]>threshold:
                predictedRes[i] = -1.0

    return predictedRes

'''
    计算弱分类器每个维度不同方向上的阈值
    寻找最佳阈值
    返回此时弱分类器相关信息
'''
def findWeakClassifyThreshold(dataMat,classLabels,weightMat):
    m,n = shape(dataMat)
    minErr = inf
    weakClassifyInfo = {}
    for dimen in range(n):                                      #对于每一个维度
        minData = dataMat[:,dimen].min()                        #找到循环范围
        maxData = dataMat[:,dimen].max()
        stepSize = 1
        stepNum = int((maxData - minData)/stepSize)
        thr = minData - stepSize
        for num in range(stepNum+1):
            thr += stepSize                                     #每次更新阈值
            for direction in ['front','back']:                  #分别判断两个方向
                predictedRes = tryThreshold(dataMat,dimen,thr,direction)
                errArr = mat(ones((m,1)))
                errArr[(predictedRes == mat(classLabels).T)] = 0#不计算分类正确的误差值
                tempErr = weightMat.T * errArr
                if tempErr < minErr:                            #保存最小误差的弱分类器信息
                    minErr = tempErr
                    res = predictedRes.copy()                   #保存分类结果，用于计算alpha
                    weakClassifyInfo['dimen'] = dimen           #保存阈值的数据维度
                    weakClassifyInfo['threshold'] = thr         #保存阈值
                    weakClassifyInfo['direction'] = direction   #保存阈值的分类方向
    alpha = float(log((1.0 - minErr)/minErr)/2.0)
    #print("alpha=",alpha)
    weakClassifyInfo['alpha'] = alpha                           #保存分类器alpha，更新下次迭代的权值
    return weakClassifyInfo,res

'''
    #根据公式Wm+1 = Wm * exp(-1 * ALPHAm* Yi * Gm(Xi)/Zm更新权值。
'''
def updateWeight(weightMat,alpha,classLabels,res):
    weightMat = multiply(weightMat,exp(-1*alpha*multiply(mat(classLabels).T,res))) 
    weightMat = weightMat/weightMat.sum()
    #print(weightMat)
    return weightMat

'''
    迭代弱分类器，得到每个弱分类器的权重alpha，并将相关信息保存在weakClassifiesArr中
'''
def trainWeakClassifies(dataMat,classLabels,classifyTimes):
    m,n = shape(dataMat)
    weightMat = mat(ones((m,1))/m)                                                  #初始权值相同
    weakClassifiesArr = []
    for i in range(classifyTimes):
        weakClassify,res = findWeakClassifyThreshold(dataMat,classLabels,weightMat) #得到弱分类器相关信息，弱分类器的分类结果
        weakClassifiesArr.append(weakClassify)
        weightMat = updateWeight(weightMat,weakClassify['alpha'],classLabels,res)   #更新权值
    return weakClassifiesArr
    
'''
    根据得到的弱分类器，组合，得出分类结果
'''
def adaBoostClassify(data,weakClassifiesArr):
    result = mat(zeros((shape(data)[0],1)))
    for i in range(len(weakClassifiesArr)):
        classResult = tryThreshold(data,weakClassifiesArr[i]['dimen'],weakClassifiesArr[i]['threshold'],weakClassifiesArr[i]['direction'])
        result += classResult * weakClassifiesArr[i]['alpha'] #sum(ALPHAi*Gi(x))
    return sign(result)

def main():
    dataMat,classLabels = createData()
    weakClassifiesArr = trainWeakClassifies(dataMat,classLabels,classifyTimes = 30)

    inputData = [3.2]
    output = adaBoostClassify(mat(inputData),weakClassifiesArr)
    print(output)

if __name__ == '__main__':
    main()
