package org.aztec.dl_common;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;
import java.util.Properties;
import java.util.Random;

import org.apache.commons.lang3.RandomUtils;
import org.aztec.dl4j.common.AritificialNerualNetworkFactory;
import org.aztec.dl4j.common.ArtificialNeuralNetwork;
import org.aztec.dl4j.common.NetworkConfiguration;
import org.aztec.dl4j.common.impl.conf.AutomaticNetwokConfiguration;
import org.aztec.dl4j.common.impl.conf.SimpleNetworkConfiguration;
import org.aztec.dl4j.common.impl.data.CSVDataFileInfo;
import org.aztec.dl4j.common.impl.data.CSVDataSource;
import org.aztec.dl4j.common.utils.TrainningUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import com.google.common.collect.Lists;

import junit.framework.TestCase;

/**
 * BP网络测试类
 */
public class BP_NetworkTest 
    extends TestCase{
	
	private static final Random random = new Random();
	private static final int inputNum = 5;
	private static final int labelNum = 5;
	private static File trainningFile = new File("test/csv/test_classfy_1.csv");
	private static File testFile = new File("test/csv/test_classfy_2.csv");
	
	public static void main(String[] args) {
		//generateData();
		train(trainningFile,testFile,null);
		//testAutomaticBuildNetwork();
	}
	
	private static void testAutomaticBuildNetwork() {

		double[] ratioRanges = new double[] {0.5,0.6};
		int[] neuronNumRanges = new int[] {1294,1296};
		File workingDir = new File("test/arbiter");
		long timeout = 100000;
		int maxCandidateNum = 1000;
		int batchSize = 50;
		int labelIndex = 0;
		int numEpochs = 50;
		try {
			Properties props = new Properties();
			props.setProperty("minibatchSize", "" + batchSize);
			DataSource ds = new CSVDataSource(new CSVDataFileInfo(trainningFile, batchSize, labelIndex, labelNum),
					new CSVDataFileInfo(testFile, batchSize, labelIndex, labelNum), props);
			AutomaticNetwokConfiguration networkConfig = new AutomaticNetwokConfiguration(
					inputNum, labelNum, ratioRanges, neuronNumRanges, workingDir, timeout, maxCandidateNum,
					ds);
			networkConfig.setBiasRanges(new double[] {0.8,0.9});
			networkConfig.setLayerNum(2);
			ArtificialNeuralNetwork ann = AritificialNerualNetworkFactory.build(networkConfig);
			ann.train((DataSetIterator)ds.trainData(), labelNum, true);
			System.out.println(ann.validate((DataSetIterator)ds.testData(), labelNum, true));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	
	/**
	 * 生成数据
	 */
	private static void generateData() {
		try {
			//testMain(args);
			File oldFile = new File("test/csv/test_classfy_1.csv");
			oldFile.delete();
			File newFile = new File("test/csv/test_classfy_2.csv");
			newFile.delete();
			generateCSVData(1000, oldFile);
			generateCSVData(1000, newFile);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/**
	 * 读取csv文件数据，并训练网络
	 * @param trainFile
	 * @param testFile
	 */
	public static void train(File trainFile,File testFile,File saveFile) {
		try {
			int batchSize = 50;

	        DataSetIterator trainIter = TrainningUtils.csvToDataSet(new CSVDataFileInfo(trainFile, batchSize, 0, labelNum));
	        DataSetIterator testIter = TrainningUtils.csvToDataSet(new CSVDataFileInfo(testFile, batchSize, 0, labelNum));
	        SimpleNetworkConfiguration snc = new SimpleNetworkConfiguration(inputNum, labelNum);

	        snc.setNeuronNums(new int[] {1295});
	        snc.setLayerNum(1);
	        //snc.setBias(0.201);
	        snc.setLearningRatio(0.5792972893331771);
	        //snc.setBiases(new double[] {0.08,0.088,1});
	        snc.setBiases(new double[] {0.8559749159474366,0.8559749159474366,0.8559749159474366});
	        snc.setMomentum(0.001);
	        snc.setNumEpochs(150);
	        ArtificialNeuralNetwork bpnn = AritificialNerualNetworkFactory.build(snc);
	        if(saveFile != null && saveFile.exists()) {
	        	bpnn.load(saveFile);
	        }
	        else {
	        	long curTime = System.currentTimeMillis();
		        bpnn.train(trainIter, snc.getNumEpochs(),true);
		        long usedTime = System.currentTimeMillis() - curTime;
		        System.out.println("Train use time:" + usedTime);
	        }
	        //1295 5x
	        Evaluation eval = bpnn.validate(testIter, labelNum,true);
	        System.out.println(eval);
	        if(saveFile != null) {
	        	bpnn.save(saveFile);
	        }
	        System.out.println(bpnn.toJson());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	/**
	 * 生成csv文件数据
	 * @param sampleSize
	 * @param csvFile
	 * @throws IOException
	 */
	private static void generateCSVData(int sampleSize,File csvFile) throws IOException {
		if(!csvFile.exists()) {
			csvFile.createNewFile();
		}
		RandomAccessFile raf = new RandomAccessFile(csvFile, "rw");
		FileChannel fc = raf.getChannel();

		double[] peopleData = new double[] {};
		for(int i = 0;i < sampleSize;i++) {
			int testNum = RandomUtils.nextInt() % labelNum;
			StringBuilder writeLine = new StringBuilder("" + testNum);
			switch(testNum) {
			case 0 :
				peopleData = generatePoorData();
				break;
			case 1 :
				peopleData = generateWriteCollarData();
				break;
			case 2 :
				peopleData = generateLeaderData();
				//labels[0][i] = 2;
				break;
			case 3 :

				peopleData = generateRichData();
				//labels[0][i] = 3;
				break;
			case 4 :

				peopleData = generateGoldenBachelorData();
				//labels[0][i] = 4;
				break;
			}
			for(int j = 0;j < peopleData.length;j++) {
				writeLine.append("," + peopleData[j]);
			}
			writeLine.append("\n");
			byte[] lineBytes = writeLine.toString().getBytes();
			ByteBuffer bb = ByteBuffer.allocate(lineBytes.length);
			bb.put(lineBytes);
			bb.flip();
			fc.write(bb);
		}
		fc.close();
	}
	
	
	/**
	 * 生成样品数据
	 * @param sampleSize
	 * @return
	 */
	private static List<double[][]> generateSampleData(int sampleSize){
		
		List<double[][]> sampleAllDatas = Lists.newArrayList();
		double[][] features = new double[sampleSize][];
		double[][] labels = new double[sampleSize][];
		//double[][] labels = new double[1][sampleSize];
		for(int i = 0;i < sampleSize;i++) {
			int testNum = RandomUtils.nextInt() % 5;
			switch(testNum) {
			case 0 :
				features[i] = generatePoorData();
				labels[i] = new double[]{1,0,0,0,0};
				//labels[0][i] = 0;
				break;
			case 1 :

				features[i] = generateWriteCollarData();
				labels[i] = new double[]{0,1,0,0,0};
				//labels[0][i] = 1;
				break;
			case 2 :

				features[i] = generateLeaderData();
				labels[i] = new double[]{0,0,1,0,0};
				//labels[0][i] = 2;
				break;
			case 3 :

				features[i] = generateRichData();
				labels[i] = new double[]{0,0,0,1,0};
				//labels[0][i] = 3;
				break;
			case 4 :

				features[i] = generateGoldenBachelorData();
				labels[i] = new double[]{0,0,0,0,1};
				//labels[0][i] = 4;
				break;
			}
			
		}
		sampleAllDatas.add(features);
		sampleAllDatas.add(labels);
		return sampleAllDatas;
		
	}
	
	/**
	 * 生成随机噪声
	 * @param base
	 * @return
	 */
	private static double getRandomNoise(double base) {
		double randomNum = random.nextDouble();
		//return randomNum;
		return base * randomNum;
		//return 0;
	}
	
	
	/**
	 * 生成吊丝数据
	 * @param base
	 * @return
	 */
	private static double[] generatePoorData() {
		//为了简化数据模样，提高训练速度，钱被简化成了一个倍数的概念。当然使用真实的人民币单位也可以，只是性能会差一点而已经
		double base = 5000;
		double deposit = base + getRandomNoise(10);
		double income = base + getRandomNoise(10);
		return new double[] {deposit,income,0d,0d,0d};
		//return new double[] {deposit,income};
	}
	

	/**
	 * 生成白领数据
	 * @return
	 */
	private static double[] generateWriteCollarData() {
		double base = 12000;
		double deposit =  base + getRandomNoise(10);
		double hasChild = 0;
		double income =  base + getRandomNoise(10);
		double hasHouse = random.nextDouble() > 0.5 ? 1 : 0;
		double hasCar = random.nextDouble() > 0.5 ? 1 : 0;
		return new double[] {deposit,income,hasHouse,hasCar,hasChild};

	}
	
	/**
	 * 生成高管数据
	 * @return
	 */
	private static double[] generateLeaderData() {
		double base = 50000;
		double deposit = base + getRandomNoise(10);
		double income =  base + getRandomNoise(10) ;
		return new double[] {deposit,income,1,1,1};
	}
	
	/**
	 * 生成有钱人数据
	 * @return
	 */
	private static double[] generateRichData() {
		double base = 100000000;
		double deposit = base + getRandomNoise(10);
		double income = base + getRandomNoise(10);
		return new double[] {deposit,income,1,1,1};
	}
	
	/**
	 * 生成钻石王老五数据
	 * @return
	 */
	private static double[] generateGoldenBachelorData() {
		double base = 100000000;
		double deposit = base + getRandomNoise(10);
		double income = base + getRandomNoise(10);
		return new double[] {deposit,income,1,1,0d};
	}
}
