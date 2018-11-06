package org.aztec.dl4j.common.impl.data;

import java.io.File;

public class CSVDataFileInfo {
	
	private File file;
	private int batchSize;
	private int labelIndex;
	private int labelNums;
	private boolean normalized = false;

	public CSVDataFileInfo() {
		// TODO Auto-generated constructor stub
	}

	public CSVDataFileInfo(File file, int batchSize, int labelIndex, int labelNums) {
		super();
		this.file = file;
		this.batchSize = batchSize;
		this.labelIndex = labelIndex;
		this.labelNums = labelNums;
	}

	public File getFile() {
		return file;
	}

	public void setFile(File file) {
		this.file = file;
	}

	public int getBatchSize() {
		return batchSize;
	}

	public void setBatchSize(int batchSize) {
		this.batchSize = batchSize;
	}

	public int getLabelIndex() {
		return labelIndex;
	}

	public void setLabelIndex(int labelIndex) {
		this.labelIndex = labelIndex;
	}

	public int getLabelNums() {
		return labelNums;
	}

	public void setLabelNums(int labelNums) {
		this.labelNums = labelNums;
	}

	public boolean isNormalized() {
		return normalized;
	}

	public void setNormalized(boolean normalized) {
		this.normalized = normalized;
	}

}
