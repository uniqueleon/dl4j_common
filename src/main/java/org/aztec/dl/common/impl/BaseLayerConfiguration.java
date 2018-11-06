package org.aztec.dl.common.impl;

import org.aztec.dl.common.LayerConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class BaseLayerConfiguration implements LayerConfiguration {
	

	protected int inputNum = 10;
    protected int outputNum = 10; // number of output classes
    protected Double momentum = null;
    protected Double bias = null;
    protected Double l2 = null;
    protected Double l1 = null;
    protected Activation activation = Activation.RELU;
    protected LossFunction lossFunction = LossFunction.NEGATIVELOGLIKELIHOOD;
    protected WeightInit weightInit = WeightInit.XAVIER;
    protected LayerType type;

	public BaseLayerConfiguration(LayerType type,int input,int output) {
		this.type = type;
		this.inputNum = input;
		this.outputNum = output;
	}
	
	public BaseLayerConfiguration(LayerType type) {
		this.type = type;
	}

	public int getOutputNum() {
		// TODO Auto-generated method stub
		return outputNum;
	}

	public int getInputNum() {
		// TODO Auto-generated method stub
		return inputNum;
	}


	public Double getBias() {
		// TODO Auto-generated method stub
		return bias;
	}

	public Double getl1() {
		// TODO Auto-generated method stub
		return l1;
	}

	public Double getl2() {
		// TODO Auto-generated method stub
		return l2;
	}

	public Activation getActiavtion() {
		// TODO Auto-generated method stub
		return activation;
	}

	public LossFunction getLossFunction() {
		// TODO Auto-generated method stub
		return lossFunction;
	}

	public WeightInit getWeightInit() {
		// TODO Auto-generated method stub
		return weightInit;
	}

	public double getMomentum() {
		return momentum;
	}

	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}

	public double getL2() {
		return l2;
	}

	public void setL2(double l2) {
		this.l2 = l2;
	}

	public double getL1() {
		return l1;
	}

	public void setL1(double l1) {
		this.l1 = l1;
	}
	
	public void setInputNum(int inputNum) {
		this.inputNum = inputNum;
	}

	public void setOutputNum(int outputNum) {
		this.outputNum = outputNum;
	}

	public void setBias(double bias) {
		this.bias = bias;
	}

	public Activation getActivation() {
		return activation;
	}

	public void setActivation(Activation activation) {
		this.activation = activation;
	}

	public void setLossFunction(LossFunction lossFunction) {
		this.lossFunction = lossFunction;
	}

	public LayerType getType() {
		// TODO Auto-generated method stub
		return type;
	}

	public void setWeightInit(WeightInit weightInit) {
		this.weightInit = weightInit;
	}

	public void setType(LayerType type) {
		this.type = type;
	}

	
}
