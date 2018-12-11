package org.aztec.dl4j.common.impl.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class CSVFileReader {
	
	private File targetFile;
	private int beginLine;
	
	

	public CSVFileReader(File tFile,int beginLine) {
		targetFile = tFile;
		this.beginLine = beginLine;
	}

	public DataSetIterator read(int batch) throws IOException {
		BufferedReader fr = new BufferedReader(new FileReader(targetFile));
		
		String readLine = fr.readLine();
		
		int lineNo = 0;
		while(readLine != null) {
			if(lineNo < beginLine) {
				lineNo++;
				continue;
			}
			System.out.println(readLine);
			lineNo++;
			readLine = fr.readLine();
		}
		return null;
	}
	
	public static void main(String[] args) {
		try {
			CSVFileReader fileReader = new CSVFileReader(new File("E:/lm/ball/bet_roll_info.csv"), 1);
			fileReader.read(10);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
