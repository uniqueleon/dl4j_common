package org.aztec.dl.common.impl;

import java.util.List;

import org.apache.commons.compress.utils.Lists;
import org.aztec.dl.common.ArtificialNeuralNetworkException;
import org.aztec.dl.common.ArtificialNeuralNetworkException.ErrorCode;
import org.aztec.dl.common.LayerConfiguration;
import org.aztec.dl.common.LayerConfiguration.LayerType;
import org.aztec.dl.common.NetworkConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public abstract class BaseNetworkConfiguration implements NetworkConfiguration {

	protected double bias = 0.1;
	protected double learnningRatio = 0.006;
	protected double momentum = 0.001;
	protected int layerNum = 3;
	protected int batchSize = 50; // batch size for each epoch
	protected int rngSeed = 123; // random number seed for reproducibility
	protected int numEpochs = 150; // number of epochs to perform
	protected double l1 = 1e-4;
	protected double l2 = 1e-4;
	protected List<LayerConfiguration> layers;

	protected int inputNum;
	protected int outputNum;
	protected int defaultNeuronNum = 10;
	protected Activation defaultActivation = Activation.RELU;
	protected Activation defaultOutputActivation = Activation.SOFTMAX;
	protected WeightInit defaultWeightInit = WeightInit.XAVIER;
	protected Activation[] activations;
	protected WeightInit[] weightInits;
	protected LossFunction lossFunction = LossFunction.NEGATIVELOGLIKELIHOOD;
	protected double[] l1s;
	protected double[] l2s;
	protected int[] neruonNums;
	protected double[] biases;

	public BaseNetworkConfiguration(int inputNum, int outputNum) {
		super();
		this.inputNum = inputNum;
		this.outputNum = outputNum;
	}

	public void init() throws ArtificialNeuralNetworkException {
		initActivations();
		initl1s();
		initl2s();
		initNeuronNums();
		initWeightInits();
		initBiases();
		this.layers = generateLayerConfigurations();
	}

	protected void initl1s() throws ArtificialNeuralNetworkException {
		if (l1s == null) {
			l1s = new double[layerNum];
			for (int i = 0; i < layerNum; i++) {
				l1s[i] = l1;
			}
		} else {
			if (l1s.length < layerNum) {
				throw new ArtificialNeuralNetworkException(" l1 data missing",
						ErrorCode.L1_REGULARIZATION_CONFIG_ERROR);
			}
		}
	}

	protected void initl2s() throws ArtificialNeuralNetworkException {
		if (l2s == null) {
			l2s = new double[layerNum];
			for (int i = 0; i < layerNum; i++) {
				l2s[i] = l2;
			}
		} else {
			if (l2s.length < layerNum) {
				throw new ArtificialNeuralNetworkException(" l2 data missing",
						ErrorCode.L2_REGULARIZATION_CONFIG_ERROR);
			}
		}
	}

	protected void initBiases() throws ArtificialNeuralNetworkException {
		if (biases == null) {
			biases = new double[layerNum];
			for (int i = 0; i < layerNum; i++) {
				biases[i] = bias;
			}
		} else {
			if (biases.length < layerNum) {
				throw new ArtificialNeuralNetworkException(" biase data missing", ErrorCode.BIAS_CONFIG_ERROR);
			}
		}
	}

	protected void initWeightInits() throws ArtificialNeuralNetworkException {
		if (weightInits == null) {
			weightInits = new WeightInit[layerNum];
			for (int i = 0; i < layerNum; i++) {
				weightInits[i] = defaultWeightInit;
			}
		} else {
			if (weightInits.length < layerNum) {
				throw new ArtificialNeuralNetworkException("weight inits data missing",
						ErrorCode.ACTIVATION_CONFIG_ERROR);
			}
		}
	}

	protected void initActivations() throws ArtificialNeuralNetworkException {
		if (activations == null) {
			activations = new Activation[layerNum];
			for (int i = 0; i < layerNum; i++) {
				if (i == layerNum - 1) {
					activations[i] = defaultOutputActivation;
				} else {
					activations[i] = defaultActivation;
				}
			}
		} else {
			if (activations.length < layerNum) {
				throw new ArtificialNeuralNetworkException("Activation data missing",
						ErrorCode.ACTIVATION_CONFIG_ERROR);
			}
		}
	}
	
	protected void initNeuronNums() throws ArtificialNeuralNetworkException {
		if (neruonNums == null) {
			
			if(layerNum > 1) {
				neruonNums = new int[layerNum - 1];
				for (int i = 0; i < layerNum - 1; i++) {
					if (i == layerNum - 1) {
						neruonNums[i] = defaultNeuronNum;
					}
				}
			}
		} else {
			if (layerNum > 1 && neruonNums.length < layerNum - 1) {
				throw new ArtificialNeuralNetworkException("neruon data missing",
						ErrorCode.NETWORK_CONFIG_ERROR);
			}
		}
	}

	protected List<LayerConfiguration> generateLayerConfigurations() {
		List<LayerConfiguration> configs = Lists.newArrayList();
		BaseLayerConfiguration lastLayerConfig = null;
		for (int i = 0; i < layerNum; i++) {

			BaseLayerConfiguration layer = new BaseLayerConfiguration(
					i != layerNum - 1 ? LayerType.DENSE : LayerType.OUTPUT,
					lastLayerConfig == null ? inputNum : lastLayerConfig.getOutputNum(),
					i != layerNum - 1 ? neruonNums[i] : outputNum);
			layer.setActivation(activations[i]);
			layer.setWeightInit(weightInits[i]);
			if (i == layerNum - 1) {
				layer.setLossFunction(lossFunction);
			}
			if (biases != null && biases.length >= layerNum + 2) {
				layer.setBias(biases[i + 1]);
			} else {
				layer.setBias(bias);
			}
			lastLayerConfig = layer;
			configs.add(layer);
		}
		return configs;
	}

	public List<LayerConfiguration> getLayers() {
		return layers;
	}

	public void setLayers(List<LayerConfiguration> layers) {
		this.layers = layers;
	}

	public int getBatchSize() {
		return batchSize;
	}

	public void setBatchSize(int batchSize) {
		this.batchSize = batchSize;
	}

	public int getRngSeed() {
		return rngSeed;
	}

	public void setRngSeed(int rngSeed) {
		this.rngSeed = rngSeed;
	}

	public int getNumEpochs() {
		return numEpochs;
	}

	public void setNumEpochs(int numEpochs) {
		this.numEpochs = numEpochs;
	}

	public int getLayerNum() {
		return layerNum;
	}

	public void setLayerNum(int layNum) {
		this.layerNum = layNum;
	}

	public double getLearningRatio() {
		return learnningRatio;
	}

	public void setLearningRatio(double learnRatio) {
		this.learnningRatio = learnRatio;
	}

	public double getMomentum() {
		return momentum;
	}

	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}

	public double getBias() {
		return bias;
	}

	public void setBias(double bias) {
		this.bias = bias;
	}

	public <T> T adapt(Class<? extends NetworkConfiguration> realClass) {
		if (realClass.isAssignableFrom(this.getClass())) {
			return (T) this;
		}
		return null;
	}

	public int[] getLayerNeuronNums() {
		// TODO Auto-generated method stub
		return null;
	}

	public double[] getl1s() {
		// TODO Auto-generated method stub
		return null;
	}

	public double[] getl2s() {
		// TODO Auto-generated method stub
		return null;
	}

	public double getL1() {
		return l1;
	}

	public void setL1(double l1) {
		this.l1 = l1;
	}

	public double getL2() {
		return l2;
	}

	public void setL2(double l2) {
		this.l2 = l2;
	}

	public double getInput() {
		// TODO Auto-generated method stub
		return inputNum;
	}

	public double getOutput() {
		// TODO Auto-generated method stub
		return outputNum;
	}

	public Activation[] getActivations() {
		return activations;
	}

	public void setActivations(Activation[] activations) {
		this.activations = activations;
	}

	public double[] getBiases() {
		return biases;
	}

	public void setBiases(double[] biases) {
		this.biases = biases;
	}

	public LossFunction getLossFunction() {
		return lossFunction;
	}

	public void setLossFunction(LossFunction lossFunction) {
		this.lossFunction = lossFunction;
	}

	public WeightInit[] getWeightInits() {
		return weightInits;
	}

	public void setWeightInits(WeightInit[] weightInits) {
		this.weightInits = weightInits;
	}

	public int getInputNum() {
		return inputNum;
	}

	public void setInputNum(int inputNum) {
		this.inputNum = inputNum;
	}

	public int getOutputNum() {
		return outputNum;
	}

	public void setOutputNum(int outputNum) {
		this.outputNum = outputNum;
	}

	public void setNeuronNums(int[] neuronNums) {
		this.neruonNums = neuronNums;
	}

	public double[] getL1s() {
		return l1s;
	}

	public void setL1s(double[] l1s) {
		this.l1s = l1s;
	}

	public double[] getL2s() {
		return l2s;
	}

	public void setL2s(double[] l2s) {
		this.l2s = l2s;
	}

}
