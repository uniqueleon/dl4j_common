package org.aztec.dl.common.impl;

import java.util.List;

import com.google.common.collect.Lists;

public class NetworkInput {
	
	private double[] features;
	private double[] lables;
	private List<String> labelNames;

	public NetworkInput(double[] features,double[] labels,List<String> lableNames) {
		this.features = features;
		this.lables = labels;
		this.labelNames = lableNames;
	}


	public double[] getFeatures() {
		return features;
	}


	public void setFeatures(double[] features) {
		this.features = features;
	}


	public double[] getLables() {
		return lables;
	}


	public void setLables(double[] lables) {
		this.lables = lables;
	}


	public List<String> getLabelNames() {
		return labelNames;
	}


	public void setLabelNames(List<String> lableNames) {
		this.labelNames = lableNames;
	}

	public static List<NetworkInput> creataInputs(double[][] features,double[][] labelDatas,List<String> labelNames){
		
		List<NetworkInput> inputs = Lists.newArrayList();
		for(int i = 0;i < features.length;i++) {
			inputs.add(new NetworkInput(features[i], labelDatas[i], labelNames));
		}
		return inputs;
	}
}
