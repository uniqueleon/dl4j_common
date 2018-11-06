package org.aztec.dl.common.impl;

import java.io.File;
import java.io.IOException;

import org.aztec.dl.common.ArtificialNeuralNetwork;
import org.aztec.dl.common.ArtificialNeuralNetworkException;
import org.aztec.dl.common.ArtificialNeuralNetworkException.ErrorCode;
import org.aztec.dl.common.NetworkConfiguration;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

public abstract class BaseNetwork implements ArtificialNeuralNetwork {

	protected Builder networkBuilder;
	protected MultiLayerNetwork network;

	public BaseNetwork() {
		// TODO Auto-generated constructor stub
	}

	protected abstract void doBuild(NetworkConfiguration networkConfig) throws ArtificialNeuralNetworkException;

	public void buildNetwork(NetworkConfiguration networkConfig) throws ArtificialNeuralNetworkException {
		networkConfig.init();
		doBuild(networkConfig);
	}

	public void train(DataSetIterator trainningDatas, int numEpochs, boolean normalized)
			throws ArtificialNeuralNetworkException {
		if (network == null) {
			throw new ArtificialNeuralNetworkException("Network not build!", ErrorCode.NETWORK_NOT_BUILD);
		}
		DataSetIterator normalizedDatas = trainningDatas;
		if (!normalized) {
			normalizedDatas = normalize(trainningDatas);
		}
		network.pretrain(normalizedDatas);
		for (int i = 0; i < numEpochs; i++) {
			network.fit(normalizedDatas);
			// preprocess(trainningDatas);
			// normalizedDatas.reset();
		}
	}

	public double[] predict(double[] features) throws ArtificialNeuralNetworkException {
		if (network == null) {
			throw new ArtificialNeuralNetworkException("Network not build!", ErrorCode.NETWORK_NOT_BUILD);
		}
		if (features != null && features.length > 0) {
			INDArray outArray = network.output(Nd4j.create(features));
			return outArray.toDoubleVector();
		}

		return null;
	}

	public void save(File file) throws IOException {
		// TODO Auto-generated method stub
		if (network != null) {
			network.save(file, true);
		}
	}

	public void load(File file) throws IOException {
		if (file != null && file.exists()) {
			network = MultiLayerNetwork.load(file, true);
		}
	}

	public Evaluation validate(DataSetIterator dataSet, int outputNum, boolean normalized)
			throws ArtificialNeuralNetworkException {
		if (network == null) {
			throw new ArtificialNeuralNetworkException("Network not build!", ErrorCode.NETWORK_NOT_BUILD);
		}
		DataSetIterator normalizedDatas = dataSet;
		if (!normalized) {
			normalizedDatas = normalize(dataSet);
		}
		Evaluation eval = new Evaluation(outputNum); // create an evaluation object with 10 possible classes
		if (normalizedDatas.resetSupported()) {
			normalizedDatas.reset();
		}
		while (normalizedDatas.hasNext()) {
			DataSet next = normalizedDatas.next();
			INDArray output = network.output(next.getFeatures()); // get the networks prediction

			eval.eval(next.getLabels(), output); // check the prediction against the true class
		}
		return eval;
	}

	private DataSetIterator normalize(DataSetIterator trainningDatas) {

		return NormalizeUtils.transform(trainningDatas);
	}

}
