package org.aztec.dl4j.common.utils;

import java.io.UnsupportedEncodingException;

import com.sun.org.apache.xml.internal.security.exceptions.Base64DecodingException;
import com.sun.org.apache.xml.internal.security.utils.Base64;

public class StringUtils {

	public StringUtils() {
		// TODO Auto-generated constructor stub	
	}
	
	public static String utf8ToBase64(String utf8Text) throws UnsupportedEncodingException, Base64DecodingException {
		String base64Text = new String(utf8Text);
		return Base64.encode(utf8Text.getBytes("UTF-8"));
	}

    public static String base64ToUtf8(String base64Code) throws UnsupportedEncodingException, Base64DecodingException {
    	String oriText = new String(Base64.decode(base64Code),"UTF-8");
    	return oriText;
    }
}
