package org.aztec.dl4j.common.impl.data;

import java.io.IOException;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

import org.aztec.dl4j.common.utils.TrainningUtils;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CSVDataSource extends ThreadGroupOptimizationDataSource implements DataSource {

	private static final Logger LOG = LoggerFactory.getLogger(CSVDataSource.class);
	private static final Map<String,DataSetIterator> cachedSource = new ConcurrentHashMap<String,DataSetIterator>(); 

	public CSVDataSource(CSVDataFileInfo trainFile, CSVDataFileInfo testFile, Properties properties)
			throws IOException, InterruptedException {
		super(trainFile, testFile, properties);
	}

	public CSVDataSource() {
		
	}
	
	private String getSuffix(boolean train) {
		return train ? "_train" : "_test";
	}
	
	private DataSetIterator getData(boolean train) {
		try {
			String sourceKey = getThreadGroupName() + getSuffix(train);
			DataSetIterator dsi = (DataSetIterator) cachedSource.get(sourceKey);
			if(dsi != null) {
				dsi.reset();
			}
			else {
				CSVDataFileInfo trainFile = (CSVDataFileInfo) (train ? getTrainMetaData() : getTestMetaData());
				dsi = TrainningUtils.csvToDataSet(trainFile);
				cachedSource.put(sourceKey, dsi);
			}
			return dsi;
		} catch (Exception e) {
			return null;
		}
	}

	public Object trainData() {
		return getData(true);
	}

	public Object testData() {
		return getData(false);
	}
}
