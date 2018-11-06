package org.aztec.dl4j.common.impl.conf;

import org.aztec.dl4j.common.NetworkConfiguration;

public class SimpleNetworkConfiguration extends BaseNetworkConfiguration implements NetworkConfiguration {
	
	public SimpleNetworkConfiguration(int inputNum,int outputNum) {
		super(inputNum,outputNum);
	}
    
	public NetworkConfigurationType getConfigType() {
		// TODO Auto-generated method stub
		return NetworkConfigurationType.SIMPLE;
	}


}
