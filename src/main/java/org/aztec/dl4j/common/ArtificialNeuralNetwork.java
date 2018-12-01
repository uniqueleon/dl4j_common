package org.aztec.dl4j.common;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public interface ArtificialNeuralNetwork {

	public void buildNetwork(NetworkConfiguration networkConfig) throws ArtificialNeuralNetworkException;
	public void train(DataSetIterator trainningDatas,int numEpochs,boolean normalized) throws ArtificialNeuralNetworkException;
	public double[] predict(double[] features)throws ArtificialNeuralNetworkException;
	public void save(File file) throws IOException;
	public void load(File file) throws IOException;
	public Evaluation validate(DataSetIterator dataSet,int outputNum,boolean normalized) throws ArtificialNeuralNetworkException;
	public String toJson();
}
