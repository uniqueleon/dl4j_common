package org.aztec.dl.common.impl;

import java.util.List;

import org.apache.commons.compress.utils.Lists;

public class NormalizationConfiguration {
	
	private List<double[]> rawDatas;
	private double[] upperLimites;

	public NormalizationConfiguration() {
		// TODO Auto-generated constructor stub
	}
	
	

	public NormalizationConfiguration(List<double[]> rawDatas, double[] upperLimites) {
		super();
		this.rawDatas = rawDatas;
		this.upperLimites = upperLimites;
	}

	public NormalizationConfiguration(double[][] rawDatas, double[] upperLimites) {
		super();
		this.rawDatas = Lists.newArrayList();
		for(int i = 0;i < rawDatas.length;i++) {
			this.rawDatas.add(rawDatas[i]);
		}
		this.upperLimites = upperLimites;
	}

	public List<double[]> getRawDatas() {
		return rawDatas;
	}

	public void setRawDatas(List<double[]> rawDatas) {
		this.rawDatas = rawDatas;
	}

	public double[] getUpperLimites() {
		return upperLimites;
	}

	public void setUpperLimites(double[] upperLimites) {
		this.upperLimites = upperLimites;
	}

	
}
