package org.aztec.dl4j.common.impl.data;

import java.io.File;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.aztec.dl4j.common.DataConvertor;

public class CSVFileConfig {

	private File targetFile;
	private String seperator;
	private int beginLine;
	private Map<Integer,DataConvertor> convertors;
	
	public CSVFileConfig(File targetFile,String seperator,
			int beginLine) {
		this.targetFile = targetFile;
		this.seperator = seperator;
		this.beginLine = beginLine;
		this.convertors = new ConcurrentHashMap<>();
	}
	
	public void addConverter(int line,DataConvertor convertor) {
		convertors.put(line, convertor);
	}

	public File getTargetFile() {
		return targetFile;
	}

	public void setTargetFile(File targetFile) {
		this.targetFile = targetFile;
	}

	public String getSeperator() {
		return seperator;
	}

	public void setSeperator(String seperator) {
		this.seperator = seperator;
	}

	public int getBeginLine() {
		return beginLine;
	}

	public void setBeginLine(int beginLine) {
		this.beginLine = beginLine;
	}

	public Map<Integer, DataConvertor> getConvertors() {
		return convertors;
	}

	public void setConvertors(Map<Integer, DataConvertor> convertors) {
		this.convertors = convertors;
	}
}
