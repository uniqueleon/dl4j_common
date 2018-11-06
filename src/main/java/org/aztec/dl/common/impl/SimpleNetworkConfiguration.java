package org.aztec.dl.common.impl;

import org.aztec.dl.common.NetworkConfiguration;

public class SimpleNetworkConfiguration extends BaseNetworkConfiguration implements NetworkConfiguration {
	
	public SimpleNetworkConfiguration(int inputNum,int outputNum) {
		super(inputNum,outputNum);
	}
    
	public NetworkConfigurationType getConfigType() {
		// TODO Auto-generated method stub
		return NetworkConfigurationType.SIMPLE;
	}


}
