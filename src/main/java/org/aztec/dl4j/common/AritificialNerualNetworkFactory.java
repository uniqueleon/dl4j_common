package org.aztec.dl4j.common;

import java.util.List;

import org.aztec.dl4j.common.impl.network.AutomaticBPNetwork;
import org.aztec.dl4j.common.impl.network.SimpleBPNN;

import com.clearspring.analytics.util.Lists;

public class AritificialNerualNetworkFactory {
	
	public static final List<CustomerizedNetwork> customerNetworks = Lists.newArrayList();

	public AritificialNerualNetworkFactory() {
		// TODO Auto-generated constructor stub
	}
	
	public static ArtificialNeuralNetwork build(NetworkConfiguration config) throws ArtificialNeuralNetworkException {
		
		if(config != null) {
			switch(config.getConfigType()) {
			case SIMPLE:
				SimpleBPNN bpnn = new SimpleBPNN();
				bpnn.buildNetwork(config);
				return bpnn;
			case AUTO:
				ArtificialNeuralNetwork network = new AutomaticBPNetwork();
				network.buildNetwork(config);
				return network;
			default :
				for(CustomerizedNetwork cNetwork : customerNetworks) {
					if(cNetwork.canBuild(config)) {
						cNetwork.buildNetwork(config);
						return cNetwork;
					}
				}
				break;
			}
		}
		return null;
		
	}

	public static void addCustomerizedNetwork(CustomerizedNetwork cNetwork) {
		
		customerNetworks.add(cNetwork);
	}
}
