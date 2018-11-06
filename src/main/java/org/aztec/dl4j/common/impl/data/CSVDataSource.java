package org.aztec.dl4j.common.impl.data;

import java.io.IOException;
import java.util.Properties;

import org.aztec.dl4j.common.utils.TrainningUtils;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CSVDataSource extends ThreadLocalOptimizationDataSource implements DataSource {

	private static final Logger LOG = LoggerFactory.getLogger(CSVDataSource.class);

	public CSVDataSource(CSVDataFileInfo trainFile, CSVDataFileInfo testFile, Properties properties)
			throws IOException, InterruptedException {
		super(TrainningUtils.csvToDataSet(trainFile), TrainningUtils.csvToDataSet(testFile), properties);
	}

	public CSVDataSource() {
		
	}
}
