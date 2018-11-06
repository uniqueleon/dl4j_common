package org.aztec.dl4j.common.impl.network;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.aztec.dl4j.common.AritificialNerualNetworkFactory;
import org.aztec.dl4j.common.ArtificialNeuralNetworkException;
import org.aztec.dl4j.common.LayerConfiguration;
import org.aztec.dl4j.common.NetworkConfiguration;
import org.aztec.dl4j.common.ArtificialNeuralNetworkException.ErrorCode;
import org.aztec.dl4j.common.impl.conf.AutomaticNetwokConfiguration;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.MultiLayerSpace.Builder;
import org.deeplearning4j.arbiter.conf.updater.SgdSpace;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.impl.EvaluationScoreFunction;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;

public class AutomaticBPNetwork extends BaseNetwork{

	public AutomaticBPNetwork() {
		// TODO Auto-generated constructor stub
	}

	public void doBuild(NetworkConfiguration networkConfig) throws ArtificialNeuralNetworkException {
		
		AutomaticNetwokConfiguration autoConfig = networkConfig.adapt(AutomaticNetwokConfiguration.class);

        ParameterSpace<Double> learningRateHyperparam = new ContinuousParameterSpace(autoConfig.getRatioRanges()[0],
        		autoConfig.getRatioRanges()[1]);  //Values will be generated uniformly at random between 0.0001 and 0.1 (inclusive)
        ParameterSpace<Integer> layerSizeHyperparam = new IntegerParameterSpace(autoConfig.getHiddenLayerNeuronNumRanges()[0],
        		autoConfig.getHiddenLayerNeuronNumRanges()[1]);            //Integer values will be generated uniformly at random between 16 and 256 (inclusive)
        Builder spaceBuilder = new MultiLayerSpace.Builder()//These next few options: fixed values for all models
                .weightInit(WeightInit.XAVIER)
                .l2(autoConfig.getL2())
                //Learning rate hyperparameter: search over different values, applied to all models
                .updater(new SgdSpace(learningRateHyperparam));
        List<LayerConfiguration> layers = networkConfig.getLayers();
        
        for(int i = 0;i < layers.size();i++) {
        	LayerConfiguration layer = layers.get(i);
        	if(i == 0) {
        		spaceBuilder = spaceBuilder.addLayer(new DenseLayerSpace.Builder()
                        .nIn(layer.getInputNum())  
                        .activation(layer.getActiavtion())
                        .nOut(layerSizeHyperparam)
                        .build());
        	}
        	else {
            	switch(layer.getType()) {
            	case DENSE:
                	spaceBuilder = spaceBuilder.addLayer(new DenseLayerSpace.Builder()
                            .activation(layer.getActiavtion())
                            .nOut(layerSizeHyperparam)
                            .build());
                	break;
            	case OUTPUT:

                	spaceBuilder = spaceBuilder.addLayer(new OutputLayerSpace.Builder()
                            .nOut(layer.getOutputNum())
                            .activation(layer.getActiavtion())
                            .lossFunction(layer.getLossFunction())
                            .build());
                	break;
            	}
        	}
        }

        MultiLayerSpace hyperparameterSpace = spaceBuilder.numEpochs(10).build();

        CandidateGenerator candidateGenerator = new RandomSearchGenerator(hyperparameterSpace, null);    

        File f = autoConfig.getWorkingDir();
        if (f.exists()) f.delete();
        f.mkdir();
        ResultSaver modelSaver = new FileModelSaver(f.getPath());
        ScoreFunction scoreFunction = new EvaluationScoreFunction(Evaluation.Metric.ACCURACY);
        TerminationCondition[] terminationConditions = {
            new MaxTimeCondition(autoConfig.getTimeout(), TimeUnit.MILLISECONDS),
            new MaxCandidatesCondition(autoConfig.getMaxCandidateNum())};

        OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
            .candidateGenerator(candidateGenerator)
            .dataSource(autoConfig.getDataSource().getClass(),autoConfig.getConfigProperties())
            .modelSaver(modelSaver)
            .scoreFunction(scoreFunction)
            .terminationConditions(terminationConditions)
            .build();

        IOptimizationRunner runner = new LocalOptimizationRunner(configuration, new MultiLayerNetworkTaskCreator());
        runner.execute();
        int indexOfBestResult = runner.bestScoreCandidateIndex();
        String s = "Best score: " + runner.bestScore() + "\n" +
                "Index of model with best score: " + runner.bestScoreCandidateIndex() + "\n" +
                "Number of configurations evaluated: " + runner.numCandidatesCompleted() + "\n";
            System.out.println(s);
        if(indexOfBestResult != -1) {

            List<ResultReference> allResults = runner.getResults();

            try {
				OptimizationResult bestResult = allResults.get(indexOfBestResult).getResult();
				network = (MultiLayerNetwork) bestResult.getResultReference().getResultModel();
			} catch (IOException e) {
				throw new ArtificialNeuralNetworkException("IO Error!", ErrorCode.IO_ERROR);
			}

            System.out.println("\n\nConfiguration of best model:\n");
            System.out.println(network.getLayerWiseConfigurations().toJson());
        }
        else {
        	System.err.println("model train fail!!!");
        }
	}
	
	

}
