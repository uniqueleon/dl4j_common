package org.aztec.dl_common;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Hello world!
 *
 */
public class App {

	private static int seed = 100;
	private static int iterations = 100;
	private static int numRows = 100;
	private static int numColumns = 100;
	private static int outputNum = 100;

	private static Logger log = LoggerFactory.getLogger(App.class);

	public static void main(String[] args) {
		try {
			autoencoderSample();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}


	private static void autoencoderSample() throws IOException {

		
		//number of rows and columns in the input pictures
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10; // number of output classes
        int batchSize = 128; // batch size for each epoch
        int rngSeed = 123; // random number seed for reproducibility
        int numEpochs = 15; // number of epochs to perform

        Nd4j.getRandom().setSeed(seed);
        //Get the DataSetIterators:
        DataSetIterator mnistTrain;
		DataSetIterator mnistTest;
		try {
			mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
			mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

			Long curTime = System.currentTimeMillis();
			System.out.println("build model..");
	        log.info("Build model....");
	        MultiLayerConfiguration conf = null;
	        
	       ListBuilder builder = new NeuralNetConfiguration.Builder()
	                .seed(rngSeed) //include a random seed for reproducibility
	                // use stochastic gradient descent as an optimization algorithm
	                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	                .updater(new Nesterovs(0.006, 0.9))
	                .l2(1e-4)
	                .list();
	       builder =  builder.layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
	                        .nIn(numRows * numColumns)
	                        .nOut(1000)
	                        //.nOut(outputNum)
	                        .activation(Activation.RELU)
	                        .weightInit(WeightInit.XAVIER)
	                        .build());
	    		   builder =  builder.layer(1, new DenseLayer.Builder() //create hidden layer 
	                        .nIn(1000)
	                        .nOut(10)
	                        .activation(Activation.RELU)
	                        .weightInit(WeightInit.XAVIER)
	                        .build());
	    		   builder =  builder.layer(2,new LossLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) 
	                        //.nIn(1000).nOut(outputNum)
	                		//.nIn(outputNum).nOut(outputNum)
	                		.activation(Activation.SOFTMAX)
	                        .weightInit(WeightInit.XAVIER)
	                        .build());
	                conf = builder.pretrain(false).backprop(true) //use backpropagation to adjust weights
	                .build();

			System.out.println("create layer..");
	        MultiLayerNetwork model = new MultiLayerNetwork(conf);
	        model.init();
	        //print the score with every 1 iteration
	        model.setListeners(new ScoreIterationListener(1));

	        //log.info("Train model....");
	        for( int i=0; i<numEpochs; i++ ){
	            model.fit(mnistTrain);
	        }


	        log.info("Evaluate model....");
	        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
	        
	        System.out.println("Traing data.............");
	        mnistTrain.reset();
	        
	        if(mnistTrain.hasNext()){
	            DataSet next = mnistTrain.next();
	            System.out.println("output:" + next);
	            INDArray featureArr = next.getFeatures();
	            System.out.println("feature col :" + featureArr.columns());
	            
	            System.out.println("lable col :" + next.getLabels().columns());
	            

	            INDArray sampleData = featureArr.get(Nd4j.create(new double[] {0}));
	            System.out.println("sample>>>>>>>>>>:");
	            System.out.println(sampleData);
	            System.out.println("sample>>>>>>>>>>:");
	            System.out.println(">>>>>>>>>>>>>>>>>>>>lables<<<<<<<<<<<<<<<<<<<<<<<<<<<");
	            System.out.println(next.getLabels());

	            System.out.println(">>>>>>>>>>>>>>>>>>>>lables<<<<<<<<<<<<<<<<<<<<<<<<<<<");
	        }

	        System.out.println("Traing data.............");
	        while(mnistTest.hasNext()){
	            DataSet next = mnistTest.next();
	            INDArray output = model.output(next.getFeatures()); //get the networks prediction

	            //System.out.println("output:" + next);
	            //System.out.println(">>>>>>>>>>>>>>>>>>>>lables<<<<<<<<<<<<<<<<<<<<<<<<<<<");
	            //System.out.println(next.getLabels());

	            //System.out.println(">>>>>>>>>>>>>>>>>>>>lables<<<<<<<<<<<<<<<<<<<<<<<<<<<");
	            //System.out.println(output);
	            eval.eval(next.getLabels(), output); //check the prediction against the true class
	        }
	        log.info(eval.stats());
	        Long usedTime = System.currentTimeMillis() - curTime;
	        System.out.println(eval.stats());
	        System.out.println("use time:" + usedTime);
	        log.info("****************Example finished********************");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
}
