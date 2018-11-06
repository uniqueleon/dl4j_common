package org.aztec.dl4j.common.utils;

import java.io.File;
import java.io.IOException;

import org.aztec.dl4j.common.impl.data.CSVDataFileInfo;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class TrainningUtils {

	public TrainningUtils() {
		// TODO Auto-generated constructor stub
	}

	public static DataSetIterator csvToDataSet(CSVDataFileInfo fileInfo)
			throws IOException, InterruptedException {

		RecordReader rr = new CSVRecordReader();
		rr.initialize(new FileSplit(fileInfo.getFile()));
		DataSetIterator dsi = new RecordReaderDataSetIterator(rr, fileInfo.getBatchSize(), fileInfo.getLabelIndex(), fileInfo.getLabelNums());
		if(!fileInfo.isNormalized()) {
			dsi = NormalizeUtils.transform(dsi);
		}
		return dsi;
	}

}
