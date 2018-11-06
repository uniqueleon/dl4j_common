package org.aztec.dl.common.impl;

import java.util.Properties;

import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class ThreadLocalOptimizationDataSource implements DataSource {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8372324498345738745L;
	private static final ThreadLocal<DataSetIterator> trainData = new ThreadLocal<DataSetIterator>();
	private static final ThreadLocal<DataSetIterator> testData = new ThreadLocal<DataSetIterator>();
    private static final ThreadLocal<Properties> configProperties = new ThreadLocal<Properties>();
    private int minibatchSize;

    public ThreadLocalOptimizationDataSource(DataSetIterator trData,DataSetIterator teData,Properties props) {
    	trainData.set(trData);
    	testData.set(teData);
    	this.configProperties.set(props);
    }
    
    public ThreadLocalOptimizationDataSource() {
    	
    }

    public void configure(Properties properties) {
        this.minibatchSize = Integer.parseInt(properties.getProperty("minibatchSize", "16"));
    }


	public Object trainData() {
        try {
            return trainData.get();

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public Object testData() {
        try {
            return testData.get();

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public Class<?> getDataType() {
        return DataSetIterator.class;
    }

	public Properties getConfigProperties() {
		return configProperties.get();
	}

	public void setConfigProperties(Properties configProperties) {
		this.configProperties.set( configProperties);
	}
    
    
}