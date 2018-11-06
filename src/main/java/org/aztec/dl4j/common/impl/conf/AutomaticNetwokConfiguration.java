package org.aztec.dl4j.common.impl.conf;

import java.io.File;
import java.util.Properties;

import org.aztec.dl4j.common.NetworkConfiguration;
import org.aztec.dl4j.common.impl.data.ThreadLocalOptimizationDataSource;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class AutomaticNetwokConfiguration extends BaseNetworkConfiguration implements NetworkConfiguration {
	

    private double[] ratioRanges = new double[] {0.0001,0.1};
    private int[] hiddenLayerNeuronNumRanges = new int[] {16,256};
    private File workingDir;
	private Long timeout;
	private Integer maxCandidateNum;
	private DataSource dataSource;
    
	public AutomaticNetwokConfiguration(int inputNum,int outputNum,double[] ratioRanges,int[] neuronNumRanges
			,File workingDir,Long timeout, Integer maxCandidateNum,
			DataSource dataSource) {
		super(inputNum,outputNum);
    	this.ratioRanges = ratioRanges;
    	this.hiddenLayerNeuronNumRanges = neuronNumRanges;
    	this.workingDir = workingDir;
    	this.timeout = timeout;
    	this.maxCandidateNum = maxCandidateNum;
    	this.dataSource = dataSource;
	}

	public AutomaticNetwokConfiguration(int inputNum,int outputNum,double[] ratioRanges,int[] neuronNumRanges
			,File workingDir,Long timeout, Integer maxCandidateNum,
			DataSetIterator trainData,DataSetIterator testData,Properties properties) {
		super(inputNum,outputNum);
    	this.ratioRanges = ratioRanges;
    	this.hiddenLayerNeuronNumRanges = neuronNumRanges;
    	this.workingDir = workingDir;
    	this.timeout = timeout;
    	this.maxCandidateNum = maxCandidateNum;
    	dataSource = new ThreadLocalOptimizationDataSource(trainData, testData, properties);
	}

	public DataSource getDataSource() {
		return dataSource;
	}

	public void setDataSource(ThreadLocalOptimizationDataSource dataSource) {
		this.dataSource = dataSource;
	}

	public double[] getRatioRanges() {
		return ratioRanges;
	}

	public void setRatioRanges(double[] ratioRanges) {
		this.ratioRanges = ratioRanges;
	}

	public int[] getHiddenLayerNeuronNumRanges() {
		return hiddenLayerNeuronNumRanges;
	}

	public void setHiddenLayerNeuronNumRanges(int[] hiddenLayerNeuronNumRanges) {
		this.hiddenLayerNeuronNumRanges = hiddenLayerNeuronNumRanges;
	}

	public File getWorkingDir() {
		return workingDir;
	}

	public void setWorkingDir(File workingDir) {
		this.workingDir = workingDir;
	}

	public Long getTimeout() {
		return timeout;
	}

	public void setTimeout(Long timeout) {
		this.timeout = timeout;
	}

	public Integer getMaxCandidateNum() {
		return maxCandidateNum;
	}

	public void setMaxCandidateNum(Integer maxCandidateNum) {
		this.maxCandidateNum = maxCandidateNum;
	}
	

	public NetworkConfigurationType getConfigType() {
		// TODO Auto-generated method stub
		return NetworkConfigurationType.AUTO;
	}

    public Properties getConfigProperties() {
    	if(ThreadLocalOptimizationDataSource.class.isAssignableFrom(dataSource.getClass())) {
    		return ((ThreadLocalOptimizationDataSource)dataSource).getConfigProperties();
    	}
		return null;
	}

   
}
