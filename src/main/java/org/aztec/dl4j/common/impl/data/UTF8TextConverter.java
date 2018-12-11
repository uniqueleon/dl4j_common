package org.aztec.dl4j.common.impl.data;

import org.aztec.dl4j.common.ArtificialNeuralNetworkException;
import org.aztec.dl4j.common.ArtificialNeuralNetworkException.ErrorCode;
import org.aztec.dl4j.common.DataConvertor;
import org.aztec.dl4j.common.utils.StringUtils;

public class UTF8TextConverter implements DataConvertor{

	public UTF8TextConverter() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public double convert(String text) throws ArtificialNeuralNetworkException {
		try {
			String base64 = StringUtils.utf8ToBase64(text);
			return new Double(base64.hashCode());
		} catch (Exception e) {
			throw new ArtificialNeuralNetworkException(e.getMessage(), ErrorCode.CONVERT_ERROR);
		}
	}

}
