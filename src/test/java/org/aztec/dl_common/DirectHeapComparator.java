package org.aztec.dl_common;

import java.nio.ByteBuffer;

import org.deeplearning4j.models.embeddings.learning.impl.elements.RandomUtils;

public class DirectHeapComparator {

	public DirectHeapComparator() {
		
		
	}
	
	public static void main(String[] args) {
		int dataSize = 10 * 1024 * 1024;
		
		ByteBuffer heapBuffer = ByteBuffer.allocate(dataSize);
		heapBuffer.put(generateData(dataSize));
		heapBuffer.flip();
		ByteBuffer directBuffer = ByteBuffer.allocateDirect(dataSize);
		directBuffer.put(generateData(dataSize));
		directBuffer.flip();
		System.out.println("Heap buffer read time:" + testMemoryReadSpeed(heapBuffer));
		System.out.println("direct buffer read time:" + testMemoryReadSpeed(directBuffer));
	}

	private static byte[] generateData(int dataSize) {
		
		byte[] dataArray = new byte[dataSize];
		for(int i = 0;i < dataArray.length;i++) {
			dataArray[i] = (byte) RandomUtils.nextInt(); 
		}
		return dataArray;
	}
	
	private static long testMemoryReadSpeed(ByteBuffer buffer) {
		long curTime = System.currentTimeMillis();
		while(buffer.hasRemaining()) {
			buffer.get();
		}
		return System.currentTimeMillis() - curTime;
	}
}
