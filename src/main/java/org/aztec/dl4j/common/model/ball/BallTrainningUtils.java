package org.aztec.dl4j.common.model.ball;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.aztec.dl4j.common.AritificialNerualNetworkFactory;
import org.aztec.dl4j.common.ArtificialNeuralNetwork;
import org.aztec.dl4j.common.impl.conf.SimpleNetworkConfiguration;
import org.aztec.dl4j.common.impl.data.SimpleTensorIterator;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import com.clearspring.analytics.util.Lists;
import com.google.common.collect.Maps;

public class BallTrainningUtils {

	public BallTrainningUtils() {
		// TODO Auto-generated constructor stub
	}
	
	public static void testBPNN() {
		try {
			DataSetIterator dsi = getTrainningData( new File("test/ball/ball_bet_roll_data.csv"),new File("test/ball/ball_bet_match_result.csv"));
			int labelNum = 2;
			int featureNum = 13;
			SimpleNetworkConfiguration snc = new SimpleNetworkConfiguration(featureNum, labelNum);

			File saveFile = new File("test/ball/bp_save.dat");

	        snc.setNeuronNums(new int[] {300});
	        snc.setLayerNum(2);
	        snc.setActivations(new Activation[] {Activation.LEAKYRELU,Activation.LEAKYRELU,Activation.SOFTMAX});
	        //snc.setBias(0.201);
	        snc.setLearningRatio(0.05);
	        //snc.setBiases(new double[] {0.08,0.088,1});
	        snc.setL1(0.1);
	        snc.setL2(0.5);
	        snc.setBiases(new double[] {0.87,0.005,0.005});
	        snc.setMomentum(0.01);
	        snc.setNumEpochs(15000);
	        ArtificialNeuralNetwork bpnn = AritificialNerualNetworkFactory.build(snc);

        	long curTime = System.currentTimeMillis();
	        bpnn.train(dsi, snc.getNumEpochs(),false);
	        long usedTime = System.currentTimeMillis() - curTime;
	        System.out.println("Train use time:" + usedTime);
	       
	        //1295 5x
	        Evaluation eval = bpnn.validate(dsi, labelNum,true);
	        System.out.println(eval);
	        //2596799,141
	        double[] result = bpnn.predict(new double[] {144.5d,0.800d,0.800d,9d,135.5d,3d,6,0d,0d,0d,0d,0d,0d});
	        double[] result2 = bpnn.predict(new double[] {142.5d,0.800d,0.800d,6d,136.5d,0d,6d,0d,0d,0d,0d,0d,0d});

	        double[] result3 = bpnn.predict(new double[] {140.5d,0.800d,0.800d,13d,127.5d,5d,8d,0d,0d,0d,0d,0d,0d});
	        System.out.println("大:"+ result[0]);
	        System.out.println("小:" + result[1]);
	        System.out.println("2-大:"+ result2[0]);
	        System.out.println("2-小:" + result2[1]);
	        System.out.println("3-大:"+ result3[0]);
	        System.out.println("3-小:" + result3[1]);
	        //bpnn.predict(features)
	        /*if(saveFile != null) {
	        	bpnn.save(saveFile);
	        }*/
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/*public static void testAutoBPNN(DataSetIterator dsi) {
		double[] ratioRanges = new double[] {0.5,0.6};
		int[] neuronNumRanges = new int[] {1294,1296};
		File workingDir = new File("test/arbiter");
		long timeout = 100000;
		int maxCandidateNum = 1000;
		int batchSize = 50;
		int labelIndex = 0;
		int numEpochs = 50;
		int labelNum = 2;
		int inputNum = 13;
		try {
			Properties props = new Properties();
			props.setProperty("minibatchSize", "" + batchSize);
			AutomaticNetwokConfiguration networkConfig = new AutomaticNetwokConfiguration(
					inputNum, labelNum, ratioRanges, neuronNumRanges, workingDir, timeout, maxCandidateNum,
					dsi);
			networkConfig.setBiasRanges(new double[] {0.8,0.9});
			networkConfig.setLayerNum(2);
			ArtificialNeuralNetwork ann = AritificialNerualNetworkFactory.build(networkConfig);
			ann.train(dsi, labelNum, true);
			System.out.println(ann.validate((DataSetIterator)ds.testData(), labelNum, true));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}*/
	
	public static void main(String[] args) {
		
		testBPNN();
	}

	public static DataSetIterator getTrainningData(File rollInfoFile, File resultFile) throws IOException {

		Map<String, Double> matchResult = readMatchData(resultFile);
		BufferedReader br = new BufferedReader(new FileReader(rollInfoFile));
		br.readLine();
		List<String> labelNames = Lists.newArrayList();
		labelNames.add("b");
		labelNames.add("s");
		Object[] matchDatas = getOneMatchData(br, matchResult,null);
		SimpleTensorIterator tItr = null;
		while (matchDatas != null) {
			if (tItr == null) {
				tItr = new SimpleTensorIterator((List<Double[]>) matchDatas[0], (List<Double[]>) matchDatas[1],
						labelNames, Integer.MAX_VALUE);
			}
			else {
				tItr.appendDataSet((List<Double[]>) matchDatas[0], (List<Double[]>) matchDatas[1]);
			}
			if(matchDatas[3] == null)
				break;
			matchDatas = getOneMatchData(br, matchResult,(String) matchDatas[3]);
		}
		return tItr;
	}

	private static Object[] getOneMatchData(BufferedReader br, Map<String, Double> matchResult,String lastLine) throws IOException {
		String lineData = lastLine == null ? br.readLine() : lastLine;
		if (lineData == null)
			return null;
		List<Double[]> features = Lists.newArrayList();
		List<Double[]> labels = Lists.newArrayList();
		int recordCount = 0;
		String gid = null;
		while (lineData != null) {
			String[] rollDatas = lineData.split(",");
			if(gid == null) {
				gid = rollDatas[0];
			}
			else if (!gid.equals(rollDatas[0])){
				return new Object[] { features, labels, recordCount ,lineData};
			}
			Double[] featureDatas = readFeature(rollDatas);
			features.add(featureDatas);
			labels.add(readLabel(gid, featureDatas[0], matchResult));
			recordCount++;
			lineData = br.readLine();
		}
		return new Object[] { features, labels, recordCount,null };
	}

	private static Double[] readFeature(String[] rollDatas) {

		Double[] features = new Double[rollDatas.length - 1];
		for (int i = 1; i < rollDatas.length; i++) {
			features[i - 1] = Double.parseDouble(rollDatas[i]);
		}
		return features;

	}

	private static Double[] readLabel(String gid, double valvePoint, Map<String, Double> resultDatas) {
		Double resultData = resultDatas.get(gid);
		return new Double[] { resultData > valvePoint ? 1d : 0d, resultData < valvePoint ? 1d : 0d };
		//return new Double[] { 1.0d,0d };
	}

	private static Map<String, Double> readMatchData(File resultFile) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(resultFile));
		Map<String, Double> retList = Maps.newHashMap();
		br.readLine();
		String readLine = br.readLine();
		while (readLine != null) {
			String[] lineDatas = readLine.split(",");
			retList.put(lineDatas[0], Double.parseDouble(lineDatas[1]));
			readLine = br.readLine();
		}
		br.close();
		return retList;
	}

}
