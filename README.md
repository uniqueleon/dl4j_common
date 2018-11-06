# dl4j_common
深度学习框架deeplearning4j的封装框架。对BP网络，卷积网(CNN)，递归神经网络(RNN)的使用和训练进行了简化，降低了初学者学习使用的难度。

下面是一个BP网络训练的示例

		try {
          //输入的特征数
          int inputNum = 5;
          //CSV文件中标签数据对应的位置
          int labelIndex = 0;
          //标签（分类）数
			    int labelNum = 5;
          //批次大小
			    int batchSize = 50;
          //读取csv文件得到训练集
	        DataSetIterator trainIter = TrainningUtils.csvToDataSet(new CSVDataFileInfo(trainFile, batchSize, labelIndex, labelNum));
	        //读取csv文件得到测试集
          DataSetIterator testIter = TrainningUtils.csvToDataSet(new CSVDataFileInfo(trainFile, batchSize, labelIndex, labelNum));
	        //创建一个简单的BP网络配置
          NetworkConfiguration snc = new SimpleNetworkConfiguration(inputNum, labelNum);
	        //设置隐层神经元个数
          snc.setNeuronNums(new int[] {1295});
          //设置网络参数
	        snc.setLayerNum(3);
	        //snc.setBias(0.201);
          //设置学习率
	        snc.setLearningRatio(0.5);
          //设置每层偏置
	        snc.setBiases(new double[] {0.08,0.088,1});
          //设置冲量
	        snc.setMomentum(0.001);
          //设置训练世代
	        snc.setNumEpochs(150);
          //通过网络配置获取对应的自动封装好的网络模型
	        ArtificialNeuralNetwork bpnn = AritificialNerualNetworkFactory.getInstance(snc);
          //构建网络
	        bpnn.buildNetwork(snc);
          //训练网络
		      bpnn.train(trainIter, snc.getNumEpochs(),false);
          //验证网络
	        bpnn.validate(testIter, labelNum,false);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		

框架封装了大量实用的网络模型，并配有相关易学易懂的demo，省去了学习dl4j api的功夫，可以集中精力于模型调优中，非常适合初学者学习使用。欢迎各路大神批评指正。
