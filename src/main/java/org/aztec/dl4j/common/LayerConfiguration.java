package org.aztec.dl4j.common;

import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public interface LayerConfiguration {
	
	public int getOutputNum();
	public int getInputNum();
	public Double getBias();
	public Double getl1();
	public Double getl2();
	public Activation getActiavtion();
	public LossFunction getLossFunction();
	public WeightInit getWeightInit();
	public LayerType getType();
	
	public static enum LayerType{
		DENSE,OUTPUT;
	}

}
