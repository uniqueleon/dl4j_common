package org.aztec.dl.common.impl;

import java.util.Iterator;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.collect.Lists;


public class SimpleTensorIterator implements DataSetIterator {
	
	
	protected List<DataSet> datas;
	protected Iterator<DataSet> trainningDatas;
	protected List<String> labels;
	protected int inputNums;
	protected DataSetPreProcessor preprocessor;
	protected int batch;
	protected boolean normalized = false;
	
	public SimpleTensorIterator() {
		datas = Lists.newArrayList();
		labels = Lists.newArrayList();
		preprocessor = new NormalizerStandardize();
	}
	
	public SimpleTensorIterator(NetworkInput input) {
		init(Lists.newArrayList(input),1);
	}

	public SimpleTensorIterator(List<NetworkInput> inputs,int batch) {
		init(inputs,batch);
	}
	
	private void init(List<NetworkInput> inputs,int batch) {
		List<DataSet> dataList = Lists.newArrayList();
		double[][] features = new double[inputs.size()][];
		double[][] labels = new double[inputs.size()][];
		for(int i = 0;i < inputs.size();i++) {
			NetworkInput input = inputs.get(i);
			inputNums = input.getFeatures().length;
			if(this.labels == null && input.getLabelNames() != null) {
				this.labels = input.getLabelNames();
			}
			features[i] = inputs.get(i).getFeatures();
			labels[i] = inputs.get(i).getLables();
		}
		dataList.add(getDataSet(features, labels));
		/*for(NetworkInput input : inputs) {
			inputNums = input.getFeatures().length;
			if(labels == null && input.getLabelNames() != null) {
				labels = input.getLabelNames();
			}
			dataList.add(getDataSet(input.getFeatures(), input.getLables()));
		}*/
		trainningDatas = dataList.iterator();
		this.batch = batch;
		datas = Lists.newArrayList();
		datas.addAll(dataList);
		preprocessor = new NormalizerStandardize();
	}
	
	private DataSet getDataSet(double[] featureRawDatas,double[] lableRawDatas) {
		
		INDArray features = Nd4j.create(featureRawDatas);
		INDArray lables = null;
		if(lableRawDatas != null) {
			lables = Nd4j.create(lableRawDatas);
		}
		return new DataSet(features, lables);
	}
	
	private DataSet getDataSet(double[][] featureRawDatas,double[][] lableRawDatas) {
		
		INDArray features = Nd4j.create(featureRawDatas);
		INDArray lables = null;
		if(lableRawDatas != null && lableRawDatas.length > 0) {
			lables = Nd4j.create(lableRawDatas);
		}
		return new DataSet(features, lables);
	}

	public boolean hasNext() {
		// TODO Auto-generated method stub
		return trainningDatas.hasNext();
	}

	public DataSet next() {
		// TODO Auto-generated method stub
		return trainningDatas.next();
	}

	public DataSet next(int num) {
		// TODO Auto-generated method stubf
		for(int i = 0;i < num;i++) {
			 trainningDatas.next();
		}
		return trainningDatas.next();
	}

	public int inputColumns() {
		// TODO Auto-generated method stub
		return inputNums;
	}

	public int totalOutcomes() {
		// TODO Auto-generated method stub
		return labels.size();
	}

	public boolean resetSupported() {
		// TODO Auto-generated method stub
		return false;
	}

	public boolean asyncSupported() {
		// TODO Auto-generated method stub
		return false;
	}

	public void reset() {
		// TODO Auto-generated method stub
		trainningDatas = datas.iterator();
	}

	public int batch() {
		// TODO Auto-generated method stub
		return batch;
	}

	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		// TODO Auto-generated method stub
		this.preprocessor = preProcessor;
	}

	public DataSetPreProcessor getPreProcessor() {
		// TODO Auto-generated method stub
		return preprocessor;
	}

	public List<String> getLabels() {
		return labels;
	}

	public boolean isNormalized() {
		return normalized;
	}

	public void setNormalized(boolean flag) {
		// TODO Auto-generated method stub
		normalized = flag;
	}

	public void addDataSet(DataSet set) {
		datas.add(set);
	}
	
}
