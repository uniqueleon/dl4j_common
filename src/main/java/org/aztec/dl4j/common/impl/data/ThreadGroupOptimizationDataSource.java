package org.aztec.dl4j.common.impl.data;

import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public abstract class ThreadGroupOptimizationDataSource implements DataSource {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8372324498345738745L;
	private static final Map<String,Object> trainingMetaData = new ConcurrentHashMap<String, Object>();
	private static final Map<String,Object> testMetaData = new ConcurrentHashMap<String, Object>();
    private static final Map<String,Properties> configProperties = new ConcurrentHashMap<String, Properties>();
    private int minibatchSize;

    public ThreadGroupOptimizationDataSource(Object trData,Object teData,Properties props) {
    	//trainData.set(trData);
    	//testData.set(teData);
    	String threadGroupName = getThreadGroupName();
    	trainingMetaData.put(threadGroupName, trData);
    	testMetaData.put(threadGroupName, teData);
    	this.configProperties.put(threadGroupName,props);
    }
    
    public ThreadGroupOptimizationDataSource() {
    	
    }

    public void configure(Properties properties) {
        this.minibatchSize = Integer.parseInt(properties.getProperty("minibatchSize", "16"));
    }


	public <T> T getTrainMetaData() {
        try {
        	
            return (T) trainingMetaData.get(getThreadGroupName());

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public <T> T getTestMetaData() {
        try {
            return (T) testMetaData.get(getThreadGroupName());

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public Class<?> getDataType() {
        return DataSetIterator.class;
    }

	public Properties getConfigProperties() {
		return configProperties.get(getThreadGroupName());
	}

    
    protected String getThreadGroupName() {
    	return Thread.currentThread().getThreadGroup().getName();
    }
}