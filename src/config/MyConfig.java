package config;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

public class MyConfig {

	  // Write a given string to a given output fileName. If file exists - delete it, and start from a fresh file 
	  public static void overwriteStringToFile (String fileName, String str) {
	    try {
	      BufferedWriter writer = Files.newBufferedWriter (Paths.get(fileName), StandardCharsets.UTF_8, StandardOpenOption.TRUNCATE_EXISTING);
	      writer.write (str);
	      writer.close ();
	    }
	    catch (IOException e) {
	  		System.out.println ("***Error: couldn't write to file " + fileName + " due to the following exception: " + e.getMessage());
	  		System.exit(0);
	    }
		}

    // Write a given string to a given output fileName. If file exists - append
    public static void writeStringToFile (String fileName, String str) {
      try {
        BufferedWriter writer = Files.newBufferedWriter (Paths.get(fileName), StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        writer.write (str);
        writer.close ();
      }
      catch (IOException e) {
    		System.out.println ("***Error: couldn't write to file " + fileName + " due to the following exception: " + e.getMessage());
    		System.exit(0);
      }
  	}

}
