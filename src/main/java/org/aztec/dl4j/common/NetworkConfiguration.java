package org.aztec.dl4j.common;

import java.util.List;

import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public interface NetworkConfiguration {

	public <T> T adapt(Class<? extends NetworkConfiguration> realClass);
	public int getRngSeed();
	public int getNumEpochs();
	public double getL1();
	public double getL2();
	public double getBias();
	public double getLearningRatio();
	public double getInput();
	public double getOutput();
	public List<LayerConfiguration> getLayers();
	public void init() throws ArtificialNeuralNetworkException;
	public NetworkConfigurationType getConfigType();
	public void setBiases(double[] biases);
	public void setActivations(Activation[] activations);
	public void setLossFunction(LossFunction lossFunction);
	public void setWeightInits(WeightInit[] weightInits);
	public void setNeuronNums(int[] neuronNums);
	public void setL1s(double[] l1s);
	public void setL2s(double[] l2s);
	
	public static enum NetworkConfigurationType{
		SIMPLE,COMPLEX,AUTO;
	}
}
