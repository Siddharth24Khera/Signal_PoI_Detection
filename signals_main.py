import os,shutil
import numpy as np
# np.set_printoptions(threshold=np.nan)
import sys
import matplotlib.pyplot as plt
import math
from scipy.ndimage.interpolation import shift


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

class Signal:
	def __init__(self,sigVals,sId):
		self.sigVals = sigVals
		self.ptsOfInt = []
		self.sId = sId

	def setPOIAndUpdateSigVal(self,ptsOfInt):
		self.ptsOfInt = np.unique(ptsOfInt)
		# print(len(self.sigVals))
		labels = np.zeros(len(self.sigVals))
		labels[self.ptsOfInt] = 1
		self.sigVals = np.transpose(np.vstack((self.sigVals,labels)))

	def printSigVals(self,lim):
		print(self.sigVals[:lim,:])

	def plotSig(self,plotFolder):
		plt.plot(self.sigVals[:,0])
		plt.ylabel("Signal Value")
		plt.xlabel("Time")
		title = str(self.sId+1)
		plt.title("Signal "+title)
		# plt.show()
		yPOI = self.sigVals[self.ptsOfInt,0]
		plt.plot(self.ptsOfInt,yPOI,'o',color='red')
		plt.savefig(os.path.join(plotFolder,title))
		plt.clf()


class InputRead:
	def __init__(self,sigFolderPath, ptsFilePath):
		self.sigFolderPath = sigFolderPath
		self.ptsFilePath = ptsFilePath
		self.signalList = []
		self.readInp()

	def getSignalList(self):
		return self.signalList

	def readInp(self):

		def fileInd(elem):
			fName = elem.split('.')[0]
			fInd = int(fName[1:])
			return fInd

		sigFileList = [f for f in os.listdir(self.sigFolderPath) if os.path.isfile(os.path.join(self.sigFolderPath,f))]
		sigFileList.sort(key=fileInd)
		# print(sigFileList)
		for i in range(len(sigFileList)):
			file = os.path.join(self.sigFolderPath, sigFileList[i])
			with open(file,'r') as inputFile:
				lines = inputFile.readlines()
				floatList = []
				for line in lines:
					line = line.strip()
					if line == '':
						break
					floatList.append(float(line))
				npVector = np.asarray(floatList)
				sig = Signal(npVector,i)
				self.signalList.append(sig)

		poiList = []
		with open(self.ptsFilePath) as inputFile:
			lines = inputFile.readlines()
			lines = lines[1:]
			for i in range(len(lines)):
				line = lines[i]
				line = line.strip()
				if line == '':
					break
				tokens = line.split(',')
				tokens = [int(x) for x in tokens]
				poiList.append(tokens)

		poiArray = np.asarray(poiList)
		poiArray = poiArray - np.ones(poiArray.shape,dtype=np.int)

		for i in range(len(self.signalList)):
			sig = self.signalList[i]
			# print(i)
			sig.setPOIAndUpdateSigVal(poiArray[:,i])

	def printAllSigs(self,lim):
		for sig in self.signalList:
			sig.printSigVals(lim)

	def plotAllSigs(self):
		plotFolder = "plots"
		if os.path.exists(plotFolder) :
			shutil.rmtree(plotFolder, ignore_errors=True)
		os.mkdir(plotFolder)
		for sig in self.signalList:
			sig.plotSig(plotFolder)


class Checker:
	def __init__(self,true_points,predicted_points,windowSize):
		self.true_points = list(true_points[0])
		self.predicted_points = list(predicted_points[0])
		self.windowSize = windowSize
		self.falseHit = 0
		self.falseHitEntries = []
		self.trueMiss = 0
		self.trueMissEntries = []
		self.calcFHitTMiss()

	def calcFHitTMiss(self):
		for i in range(len(self.predicted_points)):
			pPoint = self.predicted_points[i]
			listOfClosePoints = []
			for p in self.true_points:
				if p <= pPoint + self.windowSize/2 and p >= pPoint - self.windowSize/2 :
					listOfClosePoints.append(p)
			if len(listOfClosePoints)== 0:
				self.falseHit +=1
				self.falseHitEntries.append(pPoint)
				continue
			closestPoint = listOfClosePoints[0]
			closestDis = math.fabs(closestPoint- pPoint)
			for p in listOfClosePoints:
				if math.fabs(p - pPoint) < closestDis:
					closestDis = math.fabs(p - pPoint)
					closestPoint = p

			self.true_points.remove(closestPoint)

		self.trueMissEntries = self.true_points
		self.trueMiss = len(self.trueMissEntries)

	def getFHitTMiss(self):
		return self.falseHit, self.trueMiss

	def getEntriesFHitTMiss(self):
		return self.falseHitEntries, self.trueMissEntries

class Model:
	def __init__(self,winSize):

		# self.model = LogisticRegression()		
		# self.model = RandomForestClassifier()
		self.model = GradientBoostingClassifier(loss='deviance')
		# self.model = SVC(kernel='linear', class_weight='balanced',probability=True)
		# self.model = AdaBoostClassifier(n_estimators=50,learning_rate=1,random_state=0)

		self.winSize = winSize
		self.listOflistOfIPointsTrue = []
		self.listOflistOfIPointsPred = []

	def transform(self,data):

		data = data[:,0]
		dataMat = data.reshape(-1,1)
		# dataMat = np.fft.fft(data)
		# print(data)
		# exit()
		for i in range(self.winSize):
			xs = np.copy(data)
			xs=shift(xs, i+1, cval=np.NaN)
			# print(xs)
			# exit()
			xs = xs.reshape(-1,1)
			dataMat = np.hstack((dataMat,xs))
			xs = np.copy(data)
			xs=shift(xs, -1*(i+1), cval=np.NaN)
			xs = xs.reshape(-1,1)
			dataMat = np.hstack((dataMat,xs))
			

		# print(dataMat)
		last = dataMat.shape[0] - self.winSize
		dataMat = dataMat[self.winSize:last,:]
		firstCol = dataMat[:,0]
		dataMat = (dataMat.transpose() - dataMat.transpose()[0]).transpose()
		dataMat[:,0]=firstCol
		return dataMat

	def train(self,train_signals):
		tDataMat = self.transform(train_signals[0].sigVals[:,:-1])
		Ytrue = train_signals[0].sigVals[:,-1]
		last = Ytrue.shape[0] - self.winSize
		Ytrue = Ytrue[self.winSize:last].reshape(-1,1)
		tDataMat = np.hstack((tDataMat,Ytrue))
		# print(tDataMat.shape)
		# exit()
		for i in range(1,len(train_signals)):
			# print(sig.sigVals)
			sig = train_signals[i]
			tData = self.transform(sig.sigVals[:,:-1])
			Ytrue = sig.sigVals[:,-1]
			last = Ytrue.shape[0] - self.winSize
			Ytrue = Ytrue[self.winSize:last].reshape(-1,1)

			tData = np.hstack((tData,Ytrue))
			hit_points_indices = np.where(Ytrue==1)[0]
			hit_points_val = tData[hit_points_indices]
			# print(hit_points_val)

			upsampling = 20
			for i in range(upsampling):
				tData = np.vstack((tData,hit_points_val))
			# Ytrue = np.vstack(Ytrue,Ytrue[hit_points_indices])
			
			tDataMat = np.vstack((tDataMat,tData))
		self.model.fit(tDataMat[:,:-1],tDataMat[:,-1])
		print("Training Complete")

	def test(self,test_signals):
		for sig in test_signals:
			tSig = self.transform(sig.sigVals[:,:-1])
			Ytrue = sig.sigVals[:,-1]
			Ypred = self.model.predict(tSig)
			Ypred = np.nonzero(Ypred)
			Ytrue = np.nonzero(Ytrue)
			self.listOflistOfIPointsPred.append(Ypred)
			self.listOflistOfIPointsTrue.append(Ytrue)
			# print(Ytrue)
			# print(Ypred)

		checkingWindowSize = 100
		print("Checking Window Size = "+str(checkingWindowSize))

		for i in range(len(test_signals)):
			sig = test_signals[i]
			check = Checker(self.listOflistOfIPointsTrue[i],self.listOflistOfIPointsPred[i],checkingWindowSize)
			fHitTMiss = check.getFHitTMiss()
			print("Total True = "+str(self.listOflistOfIPointsTrue[i][0].shape[0])+ " ;Total Predict = "+str(self.listOflistOfIPointsPred[i][0].shape[0]) + " ;False Hits = " + str(fHitTMiss[0]) + " ;True Miss = " + str(fHitTMiss[1]) )


def main():
	print("Input Format : Signals Folder Path    Points of Interest File")
	if len(sys.argv) < 2:
		print("Invalid Format. Provide input file names")
		exit()
	sigFolderPath = sys.argv[1]
	ptsFilePath = sys.argv[2]

	inp = InputRead(sigFolderPath,ptsFilePath)
	# inp.printAllSigs(10)
	# inp.plotAllSigs()
	listOfSignals = inp.signalList
	totSignals = len(listOfSignals)
	validationSetSize = 5
	for i in range(int(totSignals/validationSetSize)):
		testStartIndex = validationSetSize*i 
		testEndIndex = validationSetSize*(i+1)
		test_indices = [x for x in range(testStartIndex,testEndIndex)]
		test_signals = listOfSignals[testStartIndex:testEndIndex]
		train_signals = listOfSignals.copy()
		for i in sorted(test_indices, reverse=True):
		    del train_signals[i]

		lr_model = Model(winSize=20)
		lr_model.train(train_signals)
		lr_model.test(test_signals)
		print('\n')
		# exit()

if __name__ == '__main__':
	main()