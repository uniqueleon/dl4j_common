package org.aztec.dl4j.common.impl.network;

import java.util.List;

import org.aztec.dl4j.common.LayerConfiguration;
import org.aztec.dl4j.common.NetworkConfiguration;
import org.aztec.dl4j.common.impl.conf.SimpleNetworkConfiguration;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;

public class SimpleBPNN extends BaseNetwork{


	public SimpleBPNN() {
		// TODO Auto-generated constructor stub
	}
	
	public void reset() {
		
	}
	
	public void doBuild(NetworkConfiguration rawConfig) {
		SimpleNetworkConfiguration networkConfig = rawConfig.adapt(SimpleNetworkConfiguration.class);
		networkBuilder = new NeuralNetConfiguration.Builder();
		int layNum = networkConfig.getLayerNum();
		ListBuilder listBuilder = networkBuilder.seed(networkConfig.getRngSeed()) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.biasInit(networkConfig.getBias())
                .updater(new Nesterovs(networkConfig.getLearningRatio(), networkConfig.getMomentum()))
                //.activation(Activation.ELU)
                .l1(networkConfig.getL1())
                .l2(networkConfig.getL2())
                .list();
		
		List<LayerConfiguration> layerConfigs = networkConfig.getLayers();
		for(int i = 0;i < layerConfigs.size();i++) {
			LayerConfiguration layerConfig = layerConfigs.get(i);
			switch(layerConfig.getType()) {
			case DENSE:
				listBuilder = listBuilder.layer(i,new DenseLayer.Builder() //create the first, input layer with xavier initialization
	                    .nIn(layerConfig.getInputNum())
	                    .nOut(layerConfig.getOutputNum())
	                    .biasInit(layerConfig.getBias())
	                    .activation(layerConfig.getActiavtion())
	                    .weightInit(layerConfig.getWeightInit())
	                    .build());
				break;
			case OUTPUT:
				listBuilder = listBuilder.layer(i,new OutputLayer.Builder(layerConfig.getLossFunction())//create the first, input layer with xavier initialization
	                    .nIn(layerConfig.getInputNum())
	                    .nOut(layerConfig.getOutputNum())
	                    .biasInit(layerConfig.getBias())
	                    .activation(layerConfig.getActiavtion())
	                    .weightInit(layerConfig.getWeightInit())
	                    .build());
				break;
			}
			
		}
		MultiLayerConfiguration  mlc = listBuilder.pretrain(false).backprop(true).build();
		network = new MultiLayerNetwork(mlc);

		network.init();
        //print the score with every 1 iteration
		ScoreIterationListener sil = new ScoreIterationListener();
		network.setListeners(sil);
		/*   */
	}
	
	
	
	
}
