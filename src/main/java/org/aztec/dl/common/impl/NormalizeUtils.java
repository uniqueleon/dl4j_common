package org.aztec.dl.common.impl;

import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

public class NormalizeUtils {

	private NormalizeUtils() {
	}
	
	public static INDArray getMean(INDArray features) {
		int colNum = features.columns();
		double[] means = new double[colNum];
		for(int i = 0;i < colNum;i++) {
			double[] colDatas = features.getColumn(i).toDoubleVector();
			Mean mean = new Mean();
			means[i] = mean.evaluate(colDatas);
		}
		return Nd4j.create(means);
	}

	public static INDArray getStandardDeviation(INDArray features) {
		int colNum = features.columns();
		double[] stds = new double[colNum];
		for(int i = 0;i < colNum;i++) {
			double[] colDatas = features.getColumn(i).toDoubleVector();
			StandardDeviation std = new StandardDeviation();
			stds[i] = std.evaluate(colDatas);
		}
		return Nd4j.create(stds);
	}
	
	public static  DataSetIterator transform(DataSetIterator itr) {
		NormalizerStandardize ns = new NormalizerStandardize();
		//ns.fitLabel(true);
		ns.fit(itr);
		SimpleTensorIterator tensorIterator = new SimpleTensorIterator();
		itr.reset();
		while(itr.hasNext()) {
			DataSet ds = itr.next();
			ns.transform(ds);
			tensorIterator.addDataSet(ds);
		}
		tensorIterator.reset();
		return tensorIterator;
	}
}
