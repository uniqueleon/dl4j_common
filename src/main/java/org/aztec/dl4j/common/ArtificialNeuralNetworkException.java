package org.aztec.dl4j.common;

public class ArtificialNeuralNetworkException extends Exception{
	
	public static enum ErrorCode{
		
		BIAS_CONFIG_ERROR("CONF_E_01"),
		ACTIVATION_CONFIG_ERROR("CONF_E_02"),
		WEIGHT_INIT_CONFIG_ERROR("CONF_E_03"),
		L1_REGULARIZATION_CONFIG_ERROR("CONF_E_04"),
		L2_REGULARIZATION_CONFIG_ERROR("CONF_E_05"),
		NETWORK_CONFIG_ERROR("CONF_E_06"),
		NETWORK_NOT_BUILD("NW_E_01"),
		IO_ERROR("IO_E_01"),
		CONVERT_ERROR("IO_E_02"),
		TRAIN_FAIL("NW_E_02");
		
		
		private ErrorCode(String code) {
			this.code = code;
		}

		private String code;

		public String getCode() {
			return code;
		}

		public void setCode(String code) {
			this.code = code;
		}
		
		public static ErrorCode translate(String code) {
			
			for(ErrorCode errCode : ErrorCode.values()) {
				if(errCode.getCode().equals(code)) {
					return errCode;
				}
			}
			return null;
		}
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 8924274536363739897L;
	private ErrorCode errorCode;

	public ArtificialNeuralNetworkException(String msg,ErrorCode errorCode) {
		super(msg);
		this.errorCode = errorCode;
	}

	public ArtificialNeuralNetworkException(String msg,Throwable t,ErrorCode errorCode) {
		super(msg,t);
		this.errorCode = errorCode;
	}

	
}
